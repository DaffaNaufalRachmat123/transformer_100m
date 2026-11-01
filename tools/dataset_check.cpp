// Standalone check for the pretraining data path: tokenizer -> PretrainDataset
// -> packed batch. Built by hand (no CUDA needed):
//
//   c++ -std=c++17 -O2 -I. -Ibpe -Itransformer \
//       tools/dataset_check.cpp transformer/pretrain_dataset.cpp \
//       bpe/bpe_tokenizer.cpp -o /tmp/dataset_check
//   /tmp/dataset_check data/corpus_pretrain_v8 vocab_7 512 4
//
// It asserts the two failures that used to be silent: a batch whose tail is
// zero-filled because the dataset ignored --max_seq, and a corpus read one line
// per document because the separator was missing.
#include "bpe/bpe_tokenizer.h"
#include "transformer/pretrain_dataset.h"

#include <cstdio>
#include <iostream>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    // --encode_stdin <vocab_dir|bpe.bin>: read one text per line, print the
    // token ids. Used to diff the C++ encoder against scripts/bpe_reference.py.
    if (argc > 2 && std::string(argv[1]) == "--encode_stdin") {
        BPEEncoder e;
        std::string src = argv[2];
        if (src.size() > 4 && src.substr(src.size() - 4) == ".bin") {
            if (!e.load_binary(src)) { fprintf(stderr, "load_binary failed\n"); return 1; }
        } else {
            e.load_vocab(src);
        }
        std::string line;
        while (std::getline(std::cin, line)) {
            if (line.empty()) continue;
            auto ids = e.encode(line, false, false);
            for (size_t i = 0; i < ids.size(); ++i)
                printf("%s%u", i ? " " : "", ids[i]);
            printf("\n");
        }
        return 0;
    }

    std::string corpus = (argc > 1) ? argv[1] : "data/corpus_pretrain_v8";
    std::string vocab  = (argc > 2) ? argv[2] : "vocab_7";
    uint32_t seq_len   = (argc > 3) ? (uint32_t)std::atoi(argv[3]) : 512;
    uint32_t batch     = (argc > 4) ? (uint32_t)std::atoi(argv[4]) : 4;

    BPEEncoder tok;
    tok.load_vocab(vocab);
    printf("[tok] vocab=%zu bos=%d eos=%d pad=%d think=%u\n",
           tok.vocab_size(), tok.bos_id(), tok.eos_id(), tok.pad_id(),
           tok.special_id("<|think|>"));
    if (tok.vocab_size() == 0) { printf("FAIL: empty tokenizer\n"); return 1; }

    PretrainDatasetConfig cfg;
    cfg.corpus_root = corpus;
    cfg.seq_len     = seq_len;
    cfg.batch_size  = batch;
    cfg.shard_tokens = 1u << 18;
    cfg.prefetch_shards = 16;

    PretrainDataset ds(cfg);
    ds.set_tokenize_fn([&tok](const std::string& t, std::vector<uint32_t>& out) {
        auto ids = tok.encode(t, true, true);
        out.assign(ids.begin(), ids.end());
    });
    if (!ds.init()) { printf("FAIL: dataset init\n"); return 1; }

    if (ds.current_seq_len() != seq_len) {
        printf("FAIL: dataset seq_len=%u but requested %u\n",
               ds.current_seq_len(), seq_len);
        return 1;
    }

    std::vector<uint32_t> buf((size_t)batch * seq_len, 0xFFFFFFFFu);
    if (!ds.next_batch(buf.data(), batch)) { printf("FAIL: next_batch\n"); return 1; }

    size_t unwritten = 0, zeros = 0, oob = 0;
    std::map<uint32_t, size_t> hist;
    for (uint32_t v : buf) {
        if (v == 0xFFFFFFFFu) ++unwritten;
        else if (v == 0) ++zeros;
        if (v != 0xFFFFFFFFu && v >= tok.vocab_size()) ++oob;
        hist[v]++;
    }
    printf("[batch] %u x %u = %zu tokens | unwritten=%zu zeros(BOS)=%zu "
           "out_of_range=%zu unique=%zu\n",
           batch, seq_len, buf.size(), unwritten, zeros, oob, hist.size());

    printf("[decode] first 300 chars of sample 0:\n");
    std::vector<TokenID> s0(buf.begin(), buf.begin() + std::min<size_t>(seq_len, 160));
    std::string text = tok.decode(s0, false);
    printf("%s\n", text.substr(0, 300).c_str());

    bool ok = true;
    if (unwritten) { printf("FAIL: %zu tokens never written (seq_len mismatch)\n", unwritten); ok = false; }
    if (oob)       { printf("FAIL: %zu token ids >= vocab size\n", oob); ok = false; }
    if (zeros > buf.size() / 4) {
        printf("FAIL: %.1f%% of the batch is token 0 — document boundaries are "
               "being emitted far too often\n", 100.0 * zeros / buf.size());
        ok = false;
    }
    if (hist.size() < seq_len / 4) {
        printf("FAIL: only %zu distinct token ids in the batch\n", hist.size());
        ok = false;
    }
    printf("%s\n", ok ? "DATASET CHECK OK" : "DATASET CHECK FAILED");
    return ok ? 0 : 1;
}
