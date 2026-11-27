/* Halaman mandiri, tanpa jaringan/dependensi.
 * Ubah <html data-step-delay="850"> untuk tempo default (milidetik per operasi).
 * Kontrol tempo di setiap animasi juga bisa diubah saat halaman terbuka.
 */
(() => {
  'use strict';
  const E = window.Engine, d = JSON.parse(document.getElementById('manual-data').textContent);
  const esc = s => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const fmt = v => !Number.isFinite(v) ? String(v) : (Math.abs(v) > 0 && Math.abs(v) < .00001 ? v.toExponential(5) : v.toFixed(5));
  const tok = t => d.names[t] || `ID ${t}`;
  const decode = ids => new TextDecoder('utf-8').decode(new Uint8Array(ids.flatMap(t=>d.tokenBytes[t]||[])));
  const defaultDelay = Math.max(30, Number(document.documentElement.dataset.stepDelay) || 850);
  let model = E.makeModel(d.ids, E.util.cloneP(d.P));
  let trained = E.makeModel(d.ids, E.util.cloneP(d.trainedP));
  let history = d.history.map(r => ({step:r[0], L:r[1], gn:r[2], lr:r[3], coef:r[4]}));
  let seed = d.seed, generated = null, c, batch, back, adam, revision = 0;
  function refresh() {
    c = E.forward(model, d.A);
    batch = E.lossGrad(model, [d.A,d.B], [d.mA,d.mB], d.END);
    back = E.backwardOne(model, d.A, d.POS).R;
    adam = E.adamTrace(model, [d.A,d.B], [d.mA,d.mB], d.END, 3);
  }
  refresh();
  const players = [], builders = {}, pos = d.POS, D = 4;
  let embeddingReady=false;
  const completed=new Set();
  const pill = (v, active=false, done=false) => `<span class="story-pill${active?' story-focus':''}${done?' done':''}">${esc(v)}</span>`;
  const pills = (a, index=-1, done=-1) => `<div class="story-pills">${a.map((v,i)=>pill(v,i===index,i<=done)).join('')}</div>`;
  const nums = (label, a, index=-1) => `<div class="story-num">${esc(label)} ${a.map((v,i)=>`<strong class="${i===index?'story-focus':''}">${fmt(v)}</strong>`).join(' · ')}</div>`;
  const formula = s => `<div class="story-formula">${esc(s)}</div>`;
  const frame = (title, text, equation, stage='', result='', tag='') => ({title,text,equation,stage,result,tag});
  function table(headers, rows, active=-1, cell=-1) {
    return `<div class="tw"><table><thead><tr>${headers.map(s=>`<th>${esc(s)}</th>`).join('')}</tr></thead><tbody>${rows.map((r,i)=>`<tr>${r.map((v,j)=>`<td class="${i===active&&(cell<0||j===cell)?'story-focus':''}">${esc(typeof v==='number'?fmt(v):v)}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
  }
  // Satu frame = satu operasi yang disorot. Operand, perkalian, akumulator, hasil.
  function dot(out, a, b, al='a', bl='b') {
    const f = []; let sum = 0;
    for(let i=0;i<a.length;i++) {
      const prev=sum, term=a[i]*b[i]; sum+=term;
      f.push(frame(`${out}: suku ${i+1}/${a.length}`, 'Ambil dua angka pada dimensi yang sama, kalikan, lalu tambahkan ke jumlah sementara. Angka kuning adalah pasangan yang sedang dipakai.',
        `${al}[${i}] × ${bl}[${i}] = ${fmt(a[i])} × ${fmt(b[i])} = ${fmt(term)}\njumlah = ${fmt(prev)} + ${fmt(term)} = ${fmt(sum)}`,
        nums(al,a,i)+nums(bl,b,i)+`<div class="multiply-scene">${pill(fmt(a[i]),true)}<b>×</b>${pill(fmt(b[i]),true)}<b>=</b>${pill(fmt(term),false,true)}</div>`+`<div class="multiply-scene"><span>Σ sebelumnya</span>${pill(fmt(prev))}<b>+</b>${pill(fmt(term),true)}<b>=</b>${pill(fmt(sum),false,true)}</div>`, `Jumlah sementara = ${fmt(sum)}`, 'KALI → TAMBAH'));
    }
    f.push(frame(`${out} selesai`, 'Setelah semua dimensi dijumlahkan, kita mendapatkan satu angka keluaran. Perkalian titik bukan perkalian semua pasangan sekaligus.',
      `${out} = Σ ${al}[i] × ${bl}[i] = ${fmt(sum)}`, nums(out,[sum]), `${out} = ${fmt(sum)}`, 'HASIL'));
    return f;
  }
  function project(out,x,W,input='input') {
    const f=[];
    for(let j=0;j<W[0].length;j++) f.push(...dot(`${out}[${j}]`,x,W.map(r=>r[j]),input,`kolom ${j} bobot ${out}`));
    return f;
  }
  function vectorOp(name,a,b,op='+',why='Gabungkan angka pada dimensi yang sama.') {
    return a.map((v,i)=>{
      const r=op==='×'?v*b[i]:op==='−'?v-b[i]:v+b[i];
      return frame(`${name}[${i}]`,why,`${fmt(v)} ${op} ${fmt(b[i])} = ${fmt(r)}`,nums('jalur A',a,i)+nums('jalur B',b,i),`${name}[${i}] = ${fmt(r)}`,'PER DIMENSI');
    });
  }
  function norm(x,g,name) {
    let sum=0; const f=[];
    x.forEach((v,i)=>{sum+=v*v; f.push(frame(`${name}: kuadrat dimensi ${i}`, 'Ukuran vektor dihitung dari kuadrat semua angkanya. Tanda negatif menjadi positif ketika dikuadratkan.',
      `${fmt(v)}² = ${fmt(v*v)}; Σ x² sementara = ${fmt(sum)}`,nums('x',x,i),`Σ x² = ${fmt(sum)}`,'RMSNORM · KUADRAT'));});
    const mean=sum/x.length, rs=1/Math.sqrt(mean+1e-6);
    f.push(frame(`${name}: skala bersama`,'Bagi jumlah kuadrat dengan banyak dimensi. Tambahkan epsilon agar pembagian tetap terdefinisi, lalu ambil kebalikan akar.',
      `mean = ${fmt(sum)} / ${x.length} = ${fmt(mean)}\nrs = 1 / √(${fmt(mean)} + 0.000001) = ${fmt(rs)}`,nums('x',x),`Skala = ${fmt(rs)}`,'RMSNORM · SKALA'));
    x.forEach((v,i)=>f.push(frame(`${name}[${i}]`,'Kalikan setiap angka dengan skala yang sama dan gamma yang dipelajari. Normalisasi menjaga ukuran aktivasi tetap terkendali.',
      `${fmt(v)} × ${fmt(rs)} × γ[${i}] (${fmt(g[i])}) = ${fmt(v*rs*g[i])}`,nums('x',x,i)+nums('gamma',g,i),`${name}[${i}] = ${fmt(v*rs*g[i])}`,'RMSNORM · HASIL')));
    return f;
  }
  function rotate(x,heads,p,label,reverse=false) {
    const f=[], angle=reverse?-p:p, cs=Math.cos(angle), sn=Math.sin(angle);
    for(let h=0;h<heads;h++) {
      const a=x[h*2],b=x[h*2+1];
      f.push(frame(`${label} head ${h}: sudut`,'Dengan 2 dimensi per head hanya ada satu pasangan. Posisi menentukan sudut; putaran balik menggunakan sudut negatif.',
        `θ = ${angle} / 500000^(2×0/2) = ${angle} rad\ncos θ = ${fmt(cs)}; sin θ = ${fmt(sn)}`,nums(label,x),`Posisi = ${p}`,'RoPE'));
      f.push(frame(`${label} head ${h}: komponen pertama`,'Putar pasangan angka tanpa mengubah panjang vektornya.',
        `x′0 = ${fmt(a)} × ${fmt(cs)} − ${fmt(b)} × ${fmt(sn)} = ${fmt(a*cs-b*sn)}`,nums(label,x,h*2),`x′0 = ${fmt(a*cs-b*sn)}`,'RoPE · KALI & KURANG'));
      f.push(frame(`${label} head ${h}: komponen kedua`,'Komponen kedua menggunakan pasangan angka yang sama, dengan penjumlahan.',
        `x′1 = ${fmt(a)} × ${fmt(sn)} + ${fmt(b)} × ${fmt(cs)} = ${fmt(a*sn+b*cs)}`,nums(label,x,h*2+1),`x′1 = ${fmt(a*sn+b*cs)}`,'RoPE · KALI & TAMBAH'));
    } return f;
  }
  function softmax(a,label,labels=null) {
    const f=[],mx=Math.max(...a),ex=a.map(v=>Math.exp(v-mx)),z=ex.reduce((s,v)=>s+v,0);
    f.push(frame(`${label}: cari maksimum`,'Skor belum merupakan peluang. Mengurangi semua skor dengan maksimum mencegah eksponensial terlalu besar tanpa mengubah peluang.',
      `max = ${fmt(mx)}`,nums('skor',a,a.indexOf(mx)),`Skor terbesar = ${fmt(mx)}`,'SOFTMAX · MAX'));
    let acc=0;
    ex.forEach((v,i)=>{acc+=v;f.push(frame(`${label}: eksponensial ${i}`, 'Ubah skor menjadi bilangan positif; kumpulkan semua hasil sebagai penyebut Z.',
      `e[${i}] = exp(${fmt(a[i])} − ${fmt(mx)}) = ${fmt(v)}\nZ sementara = ${fmt(acc)}`,nums('skor',a,i),`e[${i}] = ${fmt(v)}`,labels?labels[i]:'SOFTMAX · EXP'));});
    ex.forEach((v,i)=>f.push(frame(`${label}: peluang ${i}`,'Bagi masing-masing hasil dengan jumlah semuanya. Sekarang total peluang tepat 1 (100%).',
      `p[${i}] = ${fmt(v)} / ${fmt(z)} = ${fmt(v/z)}`,nums('eksponensial',ex,i),`${labels?labels[i]+': ':''}${(v/z*100).toFixed(3)}%`,'SOFTMAX · BAGI')));
    return f;
  }
  function outerFrames(name,a,b) {
    const f=[];
    for(let i=0;i<a.length;i++) for(let j=0;j<b.length;j++) f.push(frame(`${name}[${i},${j}]`,'Pada backward perkalian matriks, setiap bobot menerima hasil kali aktivasi masuk dan gradien keluar. Ini adalah outer product.',
      `${name}[${i},${j}] = ${fmt(a[i])} × ${fmt(b[j])} = ${fmt(a[i]*b[j])}`,nums('aktivasi',a,i)+nums('gradien',b,j),`${name}[${i},${j}] = ${fmt(a[i]*b[j])}`,'GRADIEN BOBOT'));
    return f;
  }
  function normBack(dy,g,nn,rs,name) {
    const terms=dy.map((v,i)=>v*g[i]*nn[i]), mean=terms.reduce((s,v)=>s+v,0)/dy.length;
    const f=dot(`${name}: Σ dy·γ·normed`,dy.map((v,i)=>v*g[i]),nn,'dy × gamma','normed');
    f.push(frame(`${name}: rata-rata koreksi`,'Mengubah satu dimensi ikut mengubah skala RMS seluruh vektor. Karena itu gradien memerlukan koreksi bersama.',
      `mean = Σ(dy·γ·normed) / ${dy.length} = ${fmt(mean)}`,nums('suku koreksi',terms),`mean = ${fmt(mean)}`,'RMSNORM MUNDUR'));
    dy.forEach((v,i)=>f.push(frame(`${name}[${i}]`,'Kurangi kontribusi perubahan skala, lalu kalikan dengan skala forward. Ini meneruskan sinyal kesalahan ke input normalisasi.',
      `dx = ${fmt(rs)} × (${fmt(v)} × ${fmt(g[i])} − ${fmt(nn[i])} × ${fmt(mean)}) = ${fmt(rs*(v*g[i]-nn[i]*mean))}\ndγ = ${fmt(v)} × ${fmt(nn[i])} = ${fmt(v*nn[i])}`,nums('dy',dy,i)+nums('normed',nn,i),`dx[${i}] = ${fmt(rs*(v*g[i]-nn[i]*mean))}`,'RMSNORM MUNDUR · HASIL')));
    return f;
  }
  // Pemutar terpisah untuk setiap proses; hanya satu timer berjalan pada satu waktu.
  let activePlayer=null, tour=false;
  function unlocked(p){const i=players.indexOf(p);return (i===0||completed.has(players[i-1].key))&&(Number(p.key.split('.')[0])<4||embeddingReady);}
  function updateFlow(){
    players.forEach(p=>p.updateLock());
    const ready=completed.has('3');
    for(const id of ['generate-weights','random-weights','restore-weights']){const button=document.getElementById(id);if(button)button.disabled=!ready;}
  }
  class Player {
    constructor(el,key,title,intro) {
      this.el=el;this.key=key;this.title=title;this.index=0;this.maxSeen=0;this.frames=null;this.rev=-1;this.timer=null;this.playing=false;
      el.innerHTML=`<h3>${esc(title)}</h3><p class="story-intro">${esc(intro)}</p>
        <div class="story-lock"></div><button class="story-go-back" type="button">Buka proses sebelumnya</button>
        <div class="story-toolbar"><button class="story-play" type="button">▶ Putar</button><button class="story-prev" type="button">← Prev</button><button class="story-next" type="button">Next →</button><button class="story-reset" type="button">↺ Ulangi</button><button class="story-continue" type="button">Proses berikutnya →</button>
        <label class="story-pace">Jeda <input type="number" min="30" max="10000" step="50" value="${defaultDelay}" aria-label="Jeda animasi ${esc(title)}"> ms</label><span class="story-counter"></span></div>
        <label class="story-seek-label">Lompat operasi <input class="story-seek" type="range" min="0" max="1" value="0" aria-label="Lompat operasi ${esc(title)}"></label>
        <div class="story-progress" role="progressbar" aria-label="Kemajuan animasi" aria-valuemin="0" aria-valuemax="100"><span></span></div>
        <div class="story-grid"><div class="story-stage"></div><div class="story-explain"><div class="story-step-tag"></div><div class="story-step-title"></div><p></p><div class="story-formula"></div><div class="story-result"></div></div></div>
        <div class="story-note">Kuning = operand aktif · Hijau = hasil. Angka ditampilkan dibulatkan; hitungan memakai presisi penuh. <span class="story-version"></span></div>`;
      this.playBtn=el.querySelector('.story-play');this.nextBtn=el.querySelector('.story-next');this.prevBtn=el.querySelector('.story-prev');
      this.playBtn.onclick=()=>this.playing?(tour=false,this.pause()):this.play();
      this.nextBtn.onclick=()=>{if(!unlocked(this))return;tour=false;this.pause();this.ensure();this.index=Math.min(this.index+1,this.frames.length-1);this.render();};
      this.prevBtn.onclick=()=>{tour=false;this.pause();this.index=Math.max(0,this.index-1);this.render();};
      el.querySelector('.story-reset').onclick=()=>{tour=false;this.pause();this.index=0;this.render();};
      this.delayInput=el.querySelector('.story-pace input');this.delayInput.onchange=()=>{this.delayInput.value=Math.max(30,Math.min(10000,Number(this.delayInput.value)||defaultDelay));};
      this.seek=el.querySelector('.story-seek');this.seek.oninput=()=>{if(!unlocked(this))return;tour=false;this.pause();this.ensure();this.index=Math.min(this.maxSeen,Number(this.seek.value));this.render();};
      el.querySelector('.story-continue').onclick=()=>{const n=players[players.indexOf(this)+1];if(n){n.el.scrollIntoView({block:'center',behavior:'smooth'});}};
      el.querySelector('.story-go-back').onclick=()=>{let n=players.find(p=>!completed.has(p.key));if(!embeddingReady&&completed.has('3'))n=players.find(p=>p.key==='4');if(n)n.el.scrollIntoView({block:'center',behavior:'smooth'});};
      players.push(this);
      this.render();
    }
    ensure(){if(this.rev!==revision){this.frames=builders[this.key]();this.rev=revision;this.index=0;}}
    updateLock(){
      const open=unlocked(this),i=players.indexOf(this);
      this.el.classList.toggle('story-locked',!open);
      this.el.querySelector('.story-lock').textContent=open?(completed.has(this.key)?'✓ Proses selesai. Bisa ditinjau ulang.':'Proses aktif · selesaikan operasi berurutan untuk membuka tahap berikutnya.'):
        (Number(this.key.split('.')[0])>=4&&!embeddingReady&&completed.has('3')?'Embedding belum dibuat. Tekan Generate embedding & bobot di tahap 4.':`Terkunci · selesaikan ${i>0?players[i-1].title:'tahap sebelumnya'} terlebih dahulu.`);
      this.el.querySelector('.story-go-back').hidden=open;
      this.playBtn.disabled=!open;this.prevBtn.disabled=!open||this.index===0;this.nextBtn.disabled=!open||this.index===this.frames.length-1;
      this.seek.disabled=!open;this.el.querySelector('.story-reset').disabled=!open;
      this.el.querySelector('.story-continue').disabled=!completed.has(this.key)||i===players.length-1;
    }
    render(){
      this.ensure();const f=this.frames[this.index];
      if(unlocked(this)){this.maxSeen=Math.max(this.maxSeen,this.index);if(this.index===this.frames.length-1)completed.add(this.key);}
      this.el.querySelector('.story-stage').innerHTML=f.stage||pills(['input','operasi','hasil'],1);
      this.el.querySelector('.story-step-title').textContent=f.title;
      this.el.querySelector('.story-step-tag').textContent=f.tag||'LANGKAH HITUNG';
      this.el.querySelector('.story-explain p').textContent=f.text;
      this.el.querySelector('.story-explain .story-formula').textContent=f.equation;
      this.el.querySelector('.story-result').textContent=f.result;
      this.el.querySelector('.story-counter').textContent=`${this.index+1} / ${this.frames.length} operasi`;
      this.seek.max=this.frames.length-1;this.seek.value=this.index;
      const percent=100*(this.index+1)/this.frames.length;
      this.el.querySelector('.story-progress span').style.width=`${percent}%`;
      this.el.querySelector('.story-progress').setAttribute('aria-valuenow',Math.round(percent));
      this.el.querySelector('.story-version').textContent=generated?`Bobot browser · seed ${seed}`:'Bobot NumPy · seed 42';
      this.prevBtn.disabled=this.index===0;this.nextBtn.disabled=this.index===this.frames.length-1;
      updateFlow();
    }
    pause(){clearTimeout(this.timer);this.timer=null;this.playing=false;this.playBtn.textContent='▶ Putar';if(activePlayer===this)activePlayer=null;}
    play(){if(!unlocked(this)){tour=false;const status=document.getElementById('tour-status');if(status)status.textContent='Tur menunggu: tekan Generate embedding & bobot di tahap 4.';return;}if(activePlayer&&activePlayer!==this)activePlayer.pause();this.ensure();if(this.index===this.frames.length-1)this.index=0;activePlayer=this;this.playing=true;this.playBtn.textContent='Ⅱ Jeda';this.render();this.schedule();}
    schedule(){this.timer=setTimeout(()=>{if(!this.playing)return;if(this.index<this.frames.length-1){this.index++;this.render();this.schedule();}else{this.pause();if(tour){const n=players.indexOf(this)+1;if(n<players.length){players[n].el.scrollIntoView({block:'center',behavior:'smooth'});players[n].play();}else{tour=false;document.getElementById('tour-status').textContent='Selesai: semua proses sudah diputar.';}}}},Math.max(30,Number(this.delayInput.value)||defaultDelay));}
  }
  function add(key,title,intro,fn,el=null){builders[key]=fn;if(!el){el=document.createElement('div');el.className='story';const sec=document.getElementById('s'+key.split('.')[0]);sec.querySelector('[data-story]').appendChild(el);}el.dataset.animation=key;return new Player(el,key,title,intro);}
  function underHeading(section,index,key,title,intro,fn){const h=document.querySelectorAll(`#s${section} > h3`)[index];const el=document.createElement('div');el.className='story';h.after(el);return add(key,title,intro,fn,el);}

  // 0–4: data → token → mask → bobot.
  add('0','Mulai dari peta proses','Ikuti perjalanan teks hingga menjadi jawaban. Setiap kotak berikutnya mempunyai animasi sendiri.',()=>[
    frame('Apa yang masuk?','Dua kalimat yang artinya sama menjadi contoh latihan dua arah. Model mini menggunakan 4 angka untuk mewakili setiap token.','EN ↔ JA → dua contoh training',pills([d.EN,d.JA],0),'Data = 2 contoh'),
    frame('Apa yang dihitung?','Teks menjadi ID, ID mengambil embedding, attention membaca konteks, FFN mengolah, lalu model menebak token berikutnya.','token → embedding → Q/K/V → attention → FFN → peluang',pills(['Token','Embedding','Attention','FFN','Peluang'],2),'Forward = membuat tebakan'),
    frame('Bagaimana belajar?','Bandingkan tebakan dengan jawaban. Backward menghitung arah perbaikan, AdamW menggeser bobot, lalu proses diulang.','loss → backward → clip → AdamW → bobot baru',pills(['Loss','Backward','Clipping','AdamW'],1),'Training = memperbaiki angka'),
    frame('Bagaimana menjawab?','Sesudah dilatih, baca prompt lalu pilih token satu demi satu sampai tanda selesai.','prompt → token 1 → token 2 → … → <|end|>',pills(['Prompt','Token baru','Token baru','Selesai'],1),'Inferensi = menghasilkan jawaban')]);
  add('1','Susun contoh latihan','Perhatikan sumber, instruksi arah, jawaban, dan penanda peran.',()=>[
    frame('Sumber dan target','Satu pasangan menyediakan jawaban yang benar. Angka belum muncul di tahap ini.','sumber Inggris → target Jepang',pills([d.EN,'→',d.JA],0),d.JA),
    frame('Beri penanda peran','System memberi konteks tugas; user berisi instruksi dan sumber; assistant berisi jawaban.','<|system|> + system + <|user|> + instruksi + sumber + <|assistant|> + jawaban + <|end|>',formula(d.textA),'Satu baris EN → JA'),
    frame('Balik arah','Model yang sama dilatih arah sebaliknya. Instruksi menentukan bahasa tujuan.',d.textB,formula(d.textB),'Satu baris JA → EN'),
    frame('Versi mini','System prompt dibuang agar contoh ringkas. BOS tetap ditambahkan di depan; jawaban dan end tetap ada.','A = BOS + user + sumber + assistant + jawaban + end',pills(d.A.map(t=>tok(t))),`${d.A.length} token dalam contoh A`)]);
  add('2','Karakter → byte → potongan → ID','Jejak merge ini berasal langsung dari tokenizer repo, bukan tokenisasi perkiraan.',()=>{
    const f=[];Array.from(d.JA).forEach(ch=>{const bytes=Array.from(new TextEncoder().encode(ch));f.push(frame(`UTF-8: ${ch}`,'Satu karakter Jepang tersimpan sebagai tiga byte. Token bisa memuat seluruh karakter atau hanya sebagian byte.',`${ch} → ${bytes.map(b=>b.toString(16).toUpperCase()).join(' ')}`,pills(bytes.map(b=>`0x${b.toString(16).toUpperCase()}`),0),`${bytes.length} byte`,'UTF-8'));});
    d.tokenTrace.forEach(t=>t.passes.forEach((a,i)=>f.push(frame(i?'Pass merge '+i:'Seed longest-match','Cari potongan terpanjang di vocab, lalu gabungkan pasangan yang tercatat di merges.txt. Ulangi hingga urutan tidak berubah.',`${t.word} → ID ${a.join(', ')}`,pills(a.map(id=>`${tok(id)} (${id})`),Math.max(0,a.length-1)),i?'Hasil pass merge':'Potongan awal','BPE REPO'))));
    d.A.forEach((t,i)=>f.push(frame(`Token posisi ${i}`,'ID adalah indeks dalam kamus vocab_9, bukan hasil rumus embedding. Untuk model mini, hanya ID unik kedua contoh dipetakan ke baris embedding.',`${tok(t)} → ID ${t} → baris E ${d.ids.indexOf(t)}`,pills(d.A.map(t=>tok(t)),i,i-1),`ID ${t} · baris mini ${d.ids.indexOf(t)}`,'LOOKUP KAMUS')));return f;
  });
  add('3','Geser target dan tentukan mask','Tebakan selalu untuk token setelah input. Hanya tebakan jawaban yang ikut loss.',()=>d.A.map((t,i)=>frame(`Posisi input ${i}`,
    d.mA[i]?'Posisi ini dilatih. Di posisi assistant, target adalah token pertama jawaban.':'Posisi ini hanya konteks: loss tidak dihitung untuk tebakan di sini.',
    `input = ${tok(t)}\ntarget = ${tok(d.A[i+1]??d.END)}\nmask baru = ${d.mA[i]} (lama = ${d.oldA[i]})`,
    pills(d.A.map(t=>tok(t)),i)+pills(['input: '+tok(t),'target: '+tok(d.A[i+1]??d.END),'mask: '+d.mA[i]],2),d.mA[i]?'Ikut loss':'Tidak ikut loss','INPUT → TARGET')));
  const weightPlayer=add('4','Buat embedding dari nol','Tekan Generate embedding & bobot, lalu ikuti u1/u2 → angka normal z → embedding E. Selesaikan tahap ini sebelum menjalankan forward.',()=>{
    const f=[], sg=E.sigmas();
    E.ORDER.forEach(k=>model.P[k].forEach((r,i)=>r.forEach((v,j)=>{
      const u=generated&&generated.U[k][i][j],z=v/sg[k];
      if(u){
        const rows=model.P[k].map((r,row)=>[k==='E'?`${tok(d.ids[row])} · ID ${d.ids[row]}`:`baris ${row}`,...r.map((x,col)=>(row<i||row===i&&col<j)?fmt(x):'belum dibuat')]);
        const stage=table(['token / baris',...r.map((_,col)=>`d${col}`)],rows,i,j+1);
        f.push(frame(`${k}[${i},${j}]: ambil uniform`,'Seed mengatur generator acak. Dua angka uniform antara 0 dan 1 adalah bahan mentah, belum embedding.',`PRNG(seed ${seed}) → u1=${fmt(u[0])}, u2=${fmt(u[1])}`,stage+pills([`u1: ${fmt(u[0])}`,`u2: ${fmt(u[1])}`],0),'Belum ada nilai embedding untuk sel ini','1 · ANGKA ACAK'));
        f.push(frame(`${k}[${i},${j}]: ubah menjadi normal`,'Box–Muller mengubah dua angka uniform menjadi z dengan distribusi normal, berpusat di nol. z bisa positif atau negatif.',`z = √(−2 ln ${fmt(u[0])}) × cos(2π × ${fmt(u[1])}) = ${fmt(z)}`,stage+formula(`u1, u2 → Box–Muller → z = ${fmt(z)}`),`z = ${fmt(z)}`,'2 · BOX–MULLER'));
        rows[i][j+1]=fmt(v);
        f.push(frame(`${k}[${i},${j}]: isi satu sel`,k==='E'?'Embedding awal = 0,02 × z. Isi sel milik token ini. Angka ini belum memiliki makna; nilainya baru dipelajari saat training.':'Skala bobot proyeksi berbeda dari embedding; bagikan z dengan akar dimensi input.',`${k}[${i},${j}] = ${fmt(sg[k])} × ${fmt(z)} = ${fmt(v)}`,table(['token / baris',...r.map((_,col)=>`d${col}`)],rows,i,j+1)+`<div class="multiply-scene">${pill(fmt(sg[k]),true)}<b>×</b>${pill(fmt(z),true)}<b>=</b>${pill(fmt(v),false,true)}</div>`,k==='E'?`E[${tok(d.ids[i])}, d${j}] = ${fmt(v)}`:`${k}[${i},${j}] = ${fmt(v)}`,'3 · SIMPAN BOBOT'));
        return;
      }
      f.push(frame(`${k}[${i},${j}]: bangkitkan angka`,generated?'Dua bilangan uniform dari PRNG Mulberry32 diubah menjadi satu bilangan normal dengan Box–Muller. Seed sama memberikan angka sama.':'Angka awal halaman dihasilkan NumPy default_rng(42). z di sini adalah angka normal yang dipakai NumPy; pasangan uniform internal NumPy tidak direkam. Tombol Generate menggunakan rumus Box–Muller yang ditampilkan.',
        u?`u1 = ${fmt(u[0])}; u2 = ${fmt(u[1])}\nz = √(−2 ln u1) cos(2π u2) = ${fmt(z)}\n${k}[${i},${j}] = σ × z = ${fmt(sg[k])} × ${fmt(z)} = ${fmt(v)}`:`z ~ N(0,1) = ${fmt(z)}\n${k}[${i},${j}] = σ × z = ${fmt(sg[k])} × ${fmt(z)} = ${fmt(v)}`,
        nums(`${k} baris ${i}`,r,j),`${k}[${i},${j}] = ${fmt(v)}`,'INISIALISASI · SEED '+seed));
    })));
    f.push(frame('Gamma RMSNorm','Gamma tidak diacak; semuanya mulai dari 1, kemudian dapat dipelajari lewat backward.','γ_attn = γ_ffn = γ_final = [1,1,1,1]',nums('gamma',model.P.ga),'Bobot awal siap','INISIALISASI'));return f;
  });
  const actions=document.createElement('div');actions.className='story-actions';actions.innerHTML='<label class="story-pace">Seed <input id="weight-seed" type="number" min="0" max="4294967295" value="43" aria-label="Seed pembangkit bobot"></label><button id="generate-weights" type="button">⟳ Generate embedding & bobot dari nol</button><button id="random-weights" type="button">🎲 Seed acak & generate</button><button id="restore-weights" type="button">Gunakan contoh NumPy 42</button><span class="story-status" id="weight-status">Embedding belum dibuat oleh Anda. Selesaikan tahap 0–3, lalu tekan Generate.</span>';weightPlayer.el.querySelector('.story-toolbar').before(actions);
  updateFlow();

  // 5: setiap subproses forward mendapat pemutar tersendiri.
  const P=()=>model.P;
  underHeading(5,0,'5.1','5.1 · Lookup embedding & RMSNorm','Token tidak langsung mempunyai makna: ID mengambil baris angka yang dipelajari.',()=>[
    frame('Ambil embedding','Embedding adalah lookup: ambil satu baris tabel bobot berdasarkan ID token. Ini bukan menghitung ID dengan rumus.',`token ${tok(d.A[pos])} · ID ${d.A[pos]} → baris ${model.idx[d.A[pos]]} → E[baris]`,nums('x0',c.x0[pos]),'4 angka embedding','LOOKUP'),...norm(c.x0[pos],P().ga,'n1')]);
  underHeading(5,1,'5.2','5.2 · Bangun Q, K, V','Q = yang dicari; K = label kecocokan; V = informasi yang dibawa. Setiap keluaran dihitung suku demi suku.',()=>[
    ...project('Q',c.n1[pos],P().Wq,'n1'),...project('K',c.n1[pos],P().Wk,'n1'),...project('V',c.n1[pos],P().Wv,'n1')]);
  underHeading(5,2,'5.3','5.3 · Putar Q dan K (RoPE)','Posisi dimasukkan sebagai putaran; V tidak diputar.',()=>[...rotate(c.qpre[pos],2,pos,'Q'),...rotate(c.kpre[pos],1,pos,'K')]);
  underHeading(5,3,'5.4','5.4 · Siapkan K/V konteks','Setiap token sebelumnya melalui lookup, RMSNorm, proyeksi K/V, dan RoPE K. Tonton semua operasi di setiap posisi.',()=>{
    const f=[];for(let j=0;j<=pos;j++)f.push(frame(`K/V posisi ${j}`,'Baca token ini, hitung vektor K/V miliknya dan simpan agar query dapat melihat konteks.',`token ${tok(d.A[j])} → E[${model.idx[d.A[j]]}] → RMSNorm → K/V`,pills(d.A.slice(0,pos+1).map(t=>tok(t)),j,j-1),'K/V posisi '+j,'KONTEKS'),...norm(c.x0[j],P().ga,`n1 posisi ${j}`),...project(`K posisi ${j}`,c.n1[j],P().Wk,'n1'),...project(`V posisi ${j}`,c.n1[j],P().Wv,'n1'),...rotate(c.kpre[j],1,j,'K'),frame('Simpan ke konteks','K sesudah RoPE dan V siap dibaca query. Kedua head Q berbagi head K/V yang sama.',`K = [${c.K[j].map(fmt)}]; V = [${c.V[j].map(fmt)}]`,nums('K',c.K[j])+nums('V',c.V[j]),`${j+1} posisi tersedia`));return f;
  });
  underHeading(5,4,'5.5','5.5 · q₀×k₀, q₁×k₁ → attention','Sorotan menunjukkan dimensi pasangan Q/K, penjumlahan skor, softmax, lalu penjumlahan V berbobot.',()=>{
    const f=[];for(let h=0;h<2;h++){
      const q=c.Q[pos].slice(h*2,h*2+2),s=c.S[h][pos].slice(0,pos+1),w=c.P[h][pos].slice(0,pos+1);
      for(let j=0;j<=pos;j++){
        f.push(...dot(`head ${h} · q·K posisi ${j}`,q,c.K[j],'q','k'));
        f.push(frame(`Head ${h}: skala skor posisi ${j}`,'Bagi hasil perkalian titik dengan akar jumlah dimensi head. Token masa depan dimask dan tidak ikut softmax.',`${fmt(q[0]*c.K[j][0]+q[1]*c.K[j][1])} / √2 = ${fmt(s[j])}`,pills(d.A.map(t=>tok(t)),j),`skor = ${fmt(s[j])}`,'SKOR / √d'));
      }
      f.push(...softmax(s,`Head ${h} bobot`,d.A.slice(0,pos+1).map(tok)));
      for(let i=0;i<2;i++)f.push(...dot(`head ${h} out[${i}]`,w,c.V.slice(0,pos+1).map(r=>r[i]),'bobot w',`V dimensi ${i}`));
    }return f;
  });
  underHeading(5,5,'5.6','5.6 · Campurkan head & jalur pintas','Head digabung, diproyeksikan Wo, lalu ditambahkan ke embedding awal.',()=>[...project('o',c.att[pos],P().Wo,'att'),...vectorOp('x_mid',c.x0[pos],c.o[pos],'+','Jalur pintas membawa informasi awal; attention menambahkan informasi konteks.')]);
  underHeading(5,6,'5.7','5.7 · RMSNorm sebelum FFN','Normalisasi kedua memakai x_mid sebagai input.',()=>norm(c.xm[pos],P().gf,'n2'));
  underHeading(5,7,'5.8','5.8 · FFN: gate, up, SiLU, perkalian','FFN mengolah token ini sendiri. Dua jalur dibangun lalu digabung per dimensi.',()=>{
    const f=[...project('gate',c.n2[pos],P().W1,'n2'),...project('up',c.n2[pos],P().W3,'n2')];
    c.up[pos].forEach((u,i)=>{
      const ex=Math.exp(-u),sig=1/(1+ex),su=u*sig,gate=c.gate[pos][i];
      f.push(frame(`SiLU dimensi ${i}: exp`,'SiLU menghaluskan angka up dengan sigmoid. Mulai dari eksponensial nilai negatif.',`exp(−${fmt(u)}) = ${fmt(ex)}`,nums('up',c.up[pos],i),`exp = ${fmt(ex)}`,'SiLU · EXP'),
        frame(`SiLU dimensi ${i}: sigmoid`,'Sigmoid adalah faktor antara 0 dan 1.',`sigmoid = 1 / (1 + ${fmt(ex)}) = ${fmt(sig)}`,nums('up',c.up[pos],i),`sigmoid = ${fmt(sig)}`,'SiLU · BAGI'),
        frame(`SiLU dimensi ${i}: hasil`,'Kalikan up dengan sigmoid-nya.',`${fmt(u)} × ${fmt(sig)} = ${fmt(su)}`,nums('up',c.up[pos],i),`SiLU = ${fmt(su)}`,'SiLU · KALI'),
        frame(`FFN dimensi ${i}`,'Gate menentukan seberapa besar hasil jalur SiLU ikut lewat.',`${fmt(gate)} × ${fmt(su)} = ${fmt(gate*su)}`,nums('gate',c.gate[pos],i)+nums('silu',c.su[pos],i),`ff[${i}] = ${fmt(gate*su)}`,'GATE × SiLU'));
    });return f;
  });
  underHeading(5,8,'5.9','5.9 · Down, residual, RMSNorm akhir','Kembalikan 8 angka FFN menjadi 4, tambahkan jalur pintas, dan normalisasi.',()=>[...project('down',c.ff[pos],P().W2,'ff'),...vectorOp('x1',c.xm[pos],c.down[pos]),...norm(c.x1[pos],P().gF,'xf')]);
  underHeading(5,9,'5.10','5.10 · Logit → peluang → loss','Semua kandidat dihitung dengan tabel embedding yang sama, kemudian dibandingkan dengan target.',()=>{
    const f=[];d.ids.forEach((t,i)=>f.push(...dot(`logit ${tok(t)}`,c.xf[pos],P().E[i],'xf',`E baris ${i}`)));
    f.push(...softmax(c.logits[pos],'Peluang token',d.ids.map(tok)));
    const t=model.idx[d.A[pos+1]],p=E.softmax(c.logits[pos])[t];
    f.push(frame('Kesalahan target','Peluang target kecil berarti loss besar. Jika peluang target mendekati 1, loss mendekati 0.',`target = ${tok(d.A[pos+1])}\nloss = −ln(${fmt(p)}) = ${fmt(-Math.log(p))}`,nums('p target',[p]),`Loss = ${fmt(-Math.log(p))}`,'CROSS-ENTROPY'));return f;
  });
  add('6','Jumlahkan loss seluruh batch','Sekarang kerjakan seluruh tebakan yang bermask pada A dan B; rata-ratakan.',()=>{
    const f=[];let sum=0,count=0;
    [d.A,d.B].forEach((seq,b)=>seq.forEach((t,i)=>{if(![d.mA,d.mB][b][i])return;const p=batch.infos[b].p[i][batch.infos[b].tg[i]],l=-Math.log(p);sum+=l;count++;
      f.push(frame(`Contoh ${b?'B':'A'} · posisi ${i}`,'Hanya mask 1 yang menyumbang. Target adalah token berikutnya, termasuk tanda end setelah jawaban.',`target ${tok(seq[i+1])}: p = ${fmt(p)}\nloss = −ln p = ${fmt(l)}\nΣ loss sementara = ${fmt(sum)}`,pills(seq.map(tok),i),`${count} tebakan · jumlah = ${fmt(sum)}`,'LOSS BATCH'));}));
    f.push(frame('Rata-rata batch','Satu angka ini menjadi tujuan training: menurunkannya berarti meningkatkan peluang jawaban yang benar.',`L = ${fmt(sum)} / ${count} = ${fmt(batch.L)}`,nums('loss batch',[batch.L]),`L = ${fmt(batch.L)}`,'RATA-RATA'));return f;
  });

  // 7: backward angka penuh untuk satu posisi, bukan gradien batch.
  underHeading(7,0,'7.1','7.1 · Sinyal salah di setiap logit','Target ingin dinaikkan; kandidat lain ingin diturunkan. Pembagi di sini 1 karena hanya satu posisi diisolasi.',()=>d.ids.map((t,i)=>frame(`Gradien logit ${tok(t)}`,i===back.tg?'Ini target: kurangi 1 dari peluang. Gradien negatif mendorong logit naik saat update.':'Ini bukan target: gradien sama dengan peluangnya, sehingga update menurunkan skornya.',
    `${fmt(back.p[i])} − ${i===back.tg?1:0} = ${fmt(back.dl[i])}`,nums('peluang',back.p,i),`dlogit = ${fmt(back.dl[i])}`,'p − ONE-HOT')));
  underHeading(7,1,'7.2','7.2 · Embedding output & gradien xf','Tabel E digunakan lagi pada output: ia menerima gradien, dan gradien diteruskan ke xf.',()=>[
    ...outerFrames('dE output',back.dl,c.xf[pos]),...project('dxf',back.dl,P().E,'dlogit')]);
  underHeading(7,2,'7.3','7.3 · RMSNorm akhir mundur','Ikuti koreksi mean dan gradien gamma.',()=>normBack(back.dxf,P().gF,c.nnf[pos],c.rsf[pos],'dx1'));
  underHeading(7,3,'7.4','7.4 · FFN mundur','Gradien W2, gate/up, W1/W3, dan input FFN dihitung dari aturan rantai.',()=>{
    const f=[...outerFrames('dW2',c.ff[pos],back.dx1),...project('dff',back.dx1,P().W2[0].map((_,j)=>P().W2.map(r=>r[j])),'dx1')];
    back.dff.forEach((v,i)=>{const su=c.su[pos][i],sg=back.sg[i],gt=c.gate[pos][i];f.push(
      frame(`SiLU′[${i}]`,'Turunan memberitahu perubahan keluaran jika input digeser sedikit.',`${fmt(su)} + ${fmt(sg)} × (1 − ${fmt(su)}) = ${fmt(back.dsilu[i])}`,nums('silu',c.su[pos],i),`SiLU′ = ${fmt(back.dsilu[i])}`,'ATURAN RANTAI'),
      frame(`dgate[${i}]`,'Jalur gate menerima gradien dikali keluaran SiLU.',`${fmt(v)} × ${fmt(su)} = ${fmt(back.dgate[i])}`,nums('dff',back.dff,i),`dgate = ${fmt(back.dgate[i])}`,'GRADIEN GATE'),
      frame(`dup[${i}]`,'Jalur up menerima gradien dikali gate dan turunan SiLU.',`${fmt(v)} × ${fmt(gt)} × ${fmt(back.dsilu[i])} = ${fmt(back.dup[i])}`,nums('dff',back.dff,i),`dup = ${fmt(back.dup[i])}`,'GRADIEN UP'));});
    f.push(...outerFrames('dW1',c.n2[pos],back.dgate),...outerFrames('dW3',c.n2[pos],back.dup));
    const transpose=W=>W[0].map((_,j)=>W.map(r=>r[j])),a=E.util.vmT(back.dgate,P().W1),b=E.util.vmT(back.dup,P().W3);
    f.push(...project('dn2 dari gate',back.dgate,transpose(P().W1),'dgate'),...project('dn2 dari up',back.dup,transpose(P().W3),'dup'),...vectorOp('dn2',a,b));return f;
  });
  underHeading(7,4,'7.5','7.5 · Residual, RMSNorm kedua, Wo mundur','Gradien dari dua jalur dijumlahkan karena input yang sama digunakan dua kali.',()=>[
    ...normBack(back.dn2,P().gf,c.nn2[pos],c.rs2[pos],'dx_mid norm'),...vectorOp('dx_mid',back.dx1,back.dxm_n),
    ...outerFrames('dWo',c.att[pos],back.dxm),...project('datt',back.dxm,P().Wo[0].map((_,j)=>P().Wo.map(r=>r[j])),'dx_mid')]);
  underHeading(7,5,'7.6','7.6 · Attention mundur','Ikuti dp, koreksi softmax, lalu kontribusi ke Q/K/V. Kedua head menjumlahkan gradien ke K/V bersama.',()=>{
    const f=[];for(let h=0;h<2;h++){
      const r=back.heads[h],q=c.Q[pos].slice(h*2,h*2+2);
      for(let j=0;j<=pos;j++)f.push(...dot(`head ${h} dp posisi ${j}`,r.dO,c.V[j],'dO','V'));
      f.push(...dot(`head ${h} D`,r.p,r.dp,'p','dp'));
      for(let j=0;j<=pos;j++){
        f.push(frame(`Head ${h} · dscore ${j}`,'Softmax menghubungkan semua peluang. Kurangi rata-rata berbobot dp agar perubahan satu skor memperhitungkan skor lain.',
          `${fmt(r.p[j])} × (${fmt(r.dp[j])} − ${fmt(r.D)}) / √2 = ${fmt(r.ds[j])}`,nums('p',r.p,j)+nums('dp',r.dp,j),`dscore = ${fmt(r.ds[j])}`,'SOFTMAX MUNDUR'));
        for(let i=0;i<2;i++)f.push(frame(`Head ${h} · posisi ${j} · dimensi ${i}`,'Satu skor mengirim gradien ke query token ini, key konteks, dan value konteks. Tanda += berarti akumulasi kontribusi.',
          `dQ[${i}] += ${fmt(r.ds[j])} × ${fmt(c.K[j][i])} = ${fmt(r.ds[j]*c.K[j][i])}\ndK[${j},${i}] += ${fmt(r.ds[j])} × ${fmt(q[i])} = ${fmt(r.ds[j]*q[i])}\ndV[${j},${i}] += ${fmt(r.p[j])} × ${fmt(r.dO[i])} = ${fmt(r.p[j]*r.dO[i])}`,nums('q',q,i)+nums('k',c.K[j],i)+nums('dO',r.dO,i),'Akumulasikan ke Q, K, V','ATTENTION · GRADIEN'));
      }
    }return f;
  });
  underHeading(7,6,'7.7','7.7 · RoPE balik, Q/K/V, RMSNorm pertama','Gradien diputar balik; kontribusi semua posisi dijumlahkan untuk bobot K/V.',()=>{
    const f=[...rotate(back.dQ,2,pos,'dQ',true)];for(let j=0;j<=pos;j++)f.push(...rotate(back.dK[j],1,j,'dK',true));
    f.push(...outerFrames('dWq',c.n1[pos],back.dQp));
    for(const [name,grad] of [['dWk',back.dKp],['dWv',back.dV]])for(let i=0;i<4;i++)for(let k=0;k<2;k++)f.push(...dot(`${name}[${i},${k}]`,c.n1.slice(0,pos+1).map(r=>r[i]),grad.map(r=>r[k]),'n1 semua posisi','grad K/V'));
    const transpose=W=>W[0].map((_,j)=>W.map(r=>r[j]));
    for(let j=0;j<=pos;j++){
      const a=E.util.vmT(back.dKp[j],P().Wk),b=E.util.vmT(back.dV[j],P().Wv),q=j===pos?E.util.vmT(back.dQp,P().Wq):[0,0,0,0];
      if(j===pos)f.push(...project('dn1 dari Q',back.dQp,transpose(P().Wq),'dQ balik'));
      f.push(...project(`dn1 ${j} dari K`,back.dKp[j],transpose(P().Wk),'dK balik'),...project(`dn1 ${j} dari V`,back.dV[j],transpose(P().Wv),'dV'),...vectorOp('dn1 K + V',a,b),...vectorOp('dn1 + Q',a.map((v,i)=>v+b[i]),q),...normBack(back.dn1[j],P().ga,c.nn1[j],c.rs1[j],`dx0 posisi ${j}`));
      if(j===pos)f.push(...vectorOp('dx0 + residual',back.dn1[j].map((v,i)=>c.rs1[j]*(v*P().ga[i]-c.nn1[j][i]*back.dn1[j].reduce((s,v,k)=>s+v*P().ga[k]*c.nn1[j][k],0)/4)),back.dxm));
    }return f;
  });
  underHeading(7,7,'7.8','7.8 · Akumulasikan dua sisi embedding','Embedding menerima gradien output dan input. Token yang berulang menambahkan beberapa kontribusi pada baris yang sama.',()=>{
    const f=[],acc=back.dE_out.map(r=>r.slice());
    for(let j=0;j<=pos;j++)for(let i=0;i<4;i++){
      const row=model.idx[d.A[j]],old=acc[row][i];acc[row][i]+=back.dx0[j][i];
      f.push(frame(`E baris ${row} · dimensi ${i}`,'Mulai dari gradien output. Tambahkan gradien lookup dari setiap kemunculan token input. Jika token berulang, baris yang sama menerima tambahan lagi.',
        `${tok(d.A[j])}: ${fmt(old)} + ${fmt(back.dx0[j][i])} = ${fmt(acc[row][i])}`,nums('gradien input',back.dx0[j],i)+nums('akumulator dE',acc[row],i),`dE[${row},${i}] = ${fmt(acc[row][i])}`,'SCATTER ADD'));
    }return f;
  });

  // 8–11: batch optimizer → training → inferensi.
  underHeading(8,0,'8.1','8.1 · Norma global & clipping','Optimizer memakai gradien seluruh batch (bagian 6), bukan gradien satu posisi di bagian 7.',()=>{
    const f=[];let s=0;for(const k in batch.G){const a=batch.G[k].flat(Infinity);a.forEach((v,i)=>{s+=v*v;f.push(frame(`Kuadrat gradien ${k} elemen ${i}`,'Semua elemen semua tensor ikut ukuran global. Kuadrat selalu positif.',`${fmt(v)}² = ${fmt(v*v)}\nΣ g² = ${fmt(s)}`,nums(k,[v]),`Jumlah kuadrat = ${fmt(s)}`,'NORMA GLOBAL'));});}
    const gn=Math.sqrt(s),coef=gn>1?1/(gn+1e-6):1;
    f.push(frame('Pilih koefisien clip','Jika norma melebihi 1, semua gradien diperkecil dengan faktor yang sama; arahnya tetap.',`||g|| = √(${fmt(s)}) = ${fmt(gn)}\ncoef = ${gn>1?'1 / (norma + 0.000001)':'1'} = ${fmt(coef)}`,nums('norma, koefisien',[gn,coef]),`coef = ${fmt(coef)}`,'CLIPPING'));
    for(const k in batch.G)batch.G[k].flat(Infinity).forEach((v,i)=>f.push(frame(`Clip ${k} elemen ${i}`,'Terapkan satu koefisien pada semua elemen gradien.',`${fmt(v)} × ${fmt(coef)} = ${fmt(v*coef)}`,nums('mentah → clip',[v,v*coef],1),`g clip = ${fmt(v*coef)}`,'SCALE GRADIENT')));return f;
  });
  underHeading(8,1,'8.2','8.2 · Jadwal learning rate','Laju langkah dibesarkan pelan, kemudian dikecilkan menurut cosine.',()=>[0,1,2,9,19,20,100,300,599].map(s=>frame(`Learning rate langkah ${s}`,'20 langkah awal warmup. Sesudahnya cosine decay dengan batas minimum 10% LR puncak.',s<20?`LR = 0.03 × (${s}+1) / 20 = ${fmt(E.sched(s))}`:`p = (${s} − 20) / 580\nLR = max(0.03 × 0.5 × (1 + cos(πp)), 0.003) = ${fmt(E.sched(s))}`,pills(['Warmup','Cosine','Batas 10%'],s<20?0:1),`LR = ${fmt(E.sched(s))}`,'JADWAL LR')));
  underHeading(8,2,'8.3','8.3 · AdamW per bobot','Tonton ingatan m/v, koreksi bias, update, dan decay untuk Wk[0,0] pada tiga langkah awal.',()=>{
    const f=[];adam.forEach((a,i)=>{
      const w=a.Wk,prev=i?adam[i-1].Wk:{m:0,v:0};
      const stage=nums('w · g mentah · g clip',[w.w,w.g,w.gc]);
      f.push(frame(`Langkah ${i}: clipping`,`Bobot yang diikuti Wk[0,0]. Gradien berasal dari batch setelah forward/backward dengan bobot langkah ini.`,`g = ${fmt(w.g)} × ${fmt(a.coef)} = ${fmt(w.gc)}`,stage,`g = ${fmt(w.gc)}`,'ADAMW · CLIP'),
        frame(`Langkah ${i}: ingatan arah m`,'m adalah rata-rata bergerak arah gradien.',`m = 0.9 × ${fmt(prev.m)} + 0.1 × ${fmt(w.gc)} = ${fmt(w.m)}`,stage,`m = ${fmt(w.m)}`,'ADAMW · m'),
        frame(`Langkah ${i}: ingatan besar v`,'v melacak rata-rata kuadrat gradien.',`v = 0.95 × ${fmt(prev.v)} + 0.05 × ${fmt(w.gc)}² = ${fmt(w.v)}`,stage,`v = ${fmt(w.v)}`,'ADAMW · v'),
        frame(`Langkah ${i}: koreksi bias`,'m dan v mulai dari 0. Koreksi mengimbangi bias awal tersebut.',`m̂ = ${fmt(w.m)} / (1 − 0.9^${i+1}) = ${fmt(w.mh)}\nv̂ = ${fmt(w.v)} / (1 − 0.95^${i+1}) = ${fmt(w.vh)}`,stage,`m̂ = ${fmt(w.mh)}; v̂ = ${fmt(w.vh)}`,'ADAMW · KOREKSI'),
        frame(`Langkah ${i}: update`,'Laju belajar dikali arah gradien yang dinormalisasi besar gradien.',`update = ${fmt(a.lr)} × ${fmt(w.mh)} / (√${fmt(w.vh)} + 1e−8) = ${fmt(w.upd)}`,stage,`update = ${fmt(w.upd)}`,'ADAMW · LANGKAH'),
        frame(`Langkah ${i}: decay`,'Weight decay menyusutkan bobot sedikit secara terpisah dari gradien. Gamma tidak memakai decay.',`decay = ${fmt(a.lr)} × 0.01 × ${fmt(w.w)} = ${fmt(w.dec)}`,stage,`decay = ${fmt(w.dec)}`,'ADAMW · DECAY'),
        frame(`Langkah ${i}: bobot baru`,'Kurangi update dan decay dari bobot lama. Forward berikutnya memakai bobot baru ini.',`w baru = ${fmt(w.w)} − ${fmt(w.upd)} − ${fmt(w.dec)} = ${fmt(w.wn)}`,nums('lama → baru',[w.w,w.wn],1),`Wk[0,0] = ${fmt(w.wn)}`,'ADAMW · HASIL'));
    });return f;
  });
  function plot(upto){const h=history.slice(0,upto+1),ys=history.map(r=>Math.log10(Math.max(r.L,1e-12))),lo=Math.min(...ys),hi=Math.max(...ys);const points=h.map(r=>`${10+r.step/599*450},${100-(Math.log10(Math.max(r.L,1e-12))-lo)/Math.max(hi-lo,1)*85}`).join(' ');return `<svg class="story-plot" viewBox="0 0 480 115" role="img" aria-label="Kurva log loss hingga langkah ${upto}"><polyline points="${points}" fill="none" stroke="#34d399" stroke-width="2"/><text x="10" y="112" fill="#8f9bc7" font-size="9">0 → ${upto} langkah · sumbu vertikal log10(loss)</text></svg>`;}
  add('9','600 putaran training','Kurva bertambah mengikuti langkah training. Tiap titik adalah loss sebelum update; lihat bagian 5–8 untuk operasi detail.',()=>{
    const f=[];history.forEach((h,i)=>{
      const phases=['Forward & loss','Backward','Clipping','AdamW'];
      phases.forEach((phase,k)=>f.push(frame(`Langkah ${i} · ${phase}`,
        ['Bobot saat ini dipakai membuat tebakan seluruh batch. Kesalahannya dihitung sebagai rata-rata loss.','Aturan rantai menghitung gradien seluruh parameter dari loss batch.','Norma global menentukan apakah gradien perlu diperkecil.','Learning rate dan ingatan m/v mengubah seluruh bobot. Kemudian kembali ke forward.'][k],
        [`L = ${fmt(h.L)}`,`||g|| = ${fmt(h.gn)}`,`coef clip = ${fmt(h.coef)}`,`LR = ${fmt(h.lr)}; w ← w − update − decay`][k],plot(i)+pills(phases,k),`Langkah ${i+1}/600 · loss ${fmt(h.L)}`,'TRAINING LOOP')));
    });return f;
  });
  add('10','Bandingkan sebelum & sesudah belajar','Perbandingan memakai hasil training dengan bobot awal yang sedang aktif.',()=>{
    const f=[],now=E.lossGrad(trained,[d.A,d.B],[d.mA,d.mB],d.END);
    [d.A,d.B].forEach((seq,b)=>seq.forEach((t,i)=>{if(![d.mA,d.mB][b][i])return;const tg=now.infos[b].tg[i],old=batch.infos[b].p[i][tg],p=now.infos[b].p[i][tg];f.push(frame(`Contoh ${b?'B':'A'} · posisi ${i}`,'Bandingkan peluang jawaban yang sama. Probabilitas tinggi sesudah training menunjukkan model menghafal contoh ini.',`target ${tok(seq[i+1])}\np awal = ${fmt(old)} → p sesudah = ${fmt(p)}\nloss ${fmt(-Math.log(old))} → ${fmt(-Math.log(p))}`,nums('peluang awal → sesudah',[old,p],1),`Peluang target = ${(p*100).toFixed(3)}%`,'SEBELUM / SESUDAH'));}));
    d.ids.forEach((t,i)=>f.push(frame(`Embedding ${tok(t)} berubah`,'Perubahan bobot adalah hasil 600 update AdamW, bukan angka acak baru.',`E awal → E setelah training`,nums('awal',P().E[i])+nums('sesudah',trained.P.E[i]),'Bobot telah dipelajari','BOBOT TERLATIH')));return f;
  });
  add('11','Inferensi: prompt → satu token → ulang','Cache K/V bertambah per token. Sesudah itu dipilih kandidat terbesar dan byte disambung menjadi teks.',()=>{
    const f=[];for(const [label,seq] of [['EN → JA',d.A],['JA → EN',d.B]]){
      const ip=seq.indexOf(d.ASSIST),prompt=seq.slice(0,ip+1),pc=E.forward(trained,prompt),r=E.generate(trained,prompt,d.END);let output=[];
      prompt.forEach((t,j)=>f.push(frame(`${label} · baca prompt ${j}`,'Baca token prompt dengan forward, simpan K sesudah RoPE dan V. Ini disebut prefill.',`token ${tok(t)}\nK = [${pc.K[j].map(fmt)}]\nV = [${pc.V[j].map(fmt)}]`,pills(prompt.map(tok),j,j-1)+nums('K',pc.K[j])+nums('V',pc.V[j]),`Cache berisi ${j+1} token`,'PREFILL KV')));
      r.trace.forEach(t=>{const rows=t.top.map(([id,l,p])=>[tok(id),l,(p*100).toFixed(3)+'%']);f.push(frame(`${label} · kandidat posisi ${t.pos}`,'Forward memakai query token terakhir dan K/V yang sudah tersimpan. FFN serta proyeksi output menghasilkan skor semua kandidat.',`logits → softmax → kandidat terbesar`,table(['token','logit','peluang'],rows,0),`${t.cache} token di cache`,'DECODE · PELUANG'));
        if(t.tok!==d.END)output.push(t.tok);
        f.push(frame(`${label} · pilih ${tok(t.tok)}`,t.tok===d.END?'Tanda end dipilih: selesai, tanda ini tidak menjadi bagian teks jawaban.':'Token dipilih dengan greedy (argmax). Tempel ke jawaban, hitung K/V token baru, lalu ulangi forward untuk token berikutnya.',`argmax(logit) → ID ${t.tok} (${tok(t.tok)})\np = ${fmt(t.p)}`,pills(output.map(tok),output.length-1),t.tok===d.END?'Jawaban selesai':'Jawaban sementara: '+decode(output),'DECODE · TOKEN'));
        if(t.tok!==d.END){const bytes=d.tokenBytes[t.tok];f.push(frame('Sambungkan byte jawaban','Decoder membaca byte milik ID yang terpilih, menyambungkan dengan token sebelumnya, lalu membaca UTF-8. Spasi dan huruf Jepang dipertahankan.',`ID ${t.tok} → ${bytes.map(b=>b.toString(16).toUpperCase().padStart(2,'0')).join(' ')}\nUTF-8 jawaban → ${decode(output)}`,pills(bytes.map(b=>`0x${b.toString(16).toUpperCase()}`),0),decode(output),'DECODE · UTF-8'));}
      });
    }return f;
  });
  add('12','Mengapa mask dan BOS penting?','Animasi ini menjelaskan rancangan eksperimen; tabel kontrol di bawah adalah run seed tetap yang tercatat saat build.',()=>[
    frame('Mask lama melewatkan kata pertama','Posisi assistant menebak kata pertama jawaban. Mask lama 0 berarti gradien tebakan ini tidak ikut belajar.',`assistant → ${tok(d.A[pos+1])}; mask lama = 0; baru = 1`,pills(['assistant',tok(d.A[pos+1]),'mask 0 → 1'],2),'Kata pertama perlu dilatih'),
    frame('BOS menyelaraskan posisi','Menghapus BOS menggeser semua posisi RoPE dan konteks dibanding latihan. Ini dapat mengubah keluaran.',`dengan BOS: posisi 0,1,2,…\ntanpa BOS: semua token bergeser −1`,pills(['BOS','user','instruksi'],0),'Format training dan inferensi harus konsisten'),
    frame('Pemblokiran repetisi','Jawaban yang benar bisa berulang. Jika token sebelumnya diblok, argmax terpaksa memilih token lain.',`ha → ha → ha; hard-block dapat melarang ha berikutnya`,pills(['ha','ha','ha'],1),'Uji perubahan satu variabel'),
    frame('Ubah sumber','Keberhasilan pada dua contoh latihan tidak membuktikan kemampuan menerjemahkan sumber baru.',`cat → house; apakah jawaban berubah sesuai?`,pills(['cat','→','house'],2),'Evaluasi pada data baru')]);
  add('13','Dari mini ke model besar','Rumus tetap sama; skala data dan parameter bertambah.',()=>[
    frame('Dimensi bertambah','Di model mini, satu dot product hanya empat perkalian. Model besar memakai jauh lebih banyak dimensi dan head.',`D mini = 4 → D nyata = 1280`,pills(['4 dimensi','1280 dimensi'],1),'Lebih banyak angka per token'),
    frame('Layer bertambah','Keluaran satu blok menjadi input blok berikutnya. Attention dan FFN diulang di setiap layer.',`1 layer → 24 layer`,pills(['Attention','FFN','→ blok berikutnya'],2),'Struktur proses sama'),
    frame('Data bertambah','Dua contoh cukup untuk hafalan. Kemampuan umum harus diukur pada kalimat yang tidak dipakai training.',`training → evaluasi data terpisah`,pills(['Training','Data uji','Kualitas'],1),'Akhir perjalanan: ukur hasil')]);

  // Regenerate: hasil interaktif menjadi sumber angka aktif; tabel build tetap sebagai arsip.
  function archiveStatic(){
    for(let i=4;i<=11;i++){
      const sec=document.getElementById('s'+i);if(sec.querySelector(':scope > .snapshot'))continue;
      const snapshot=document.createElement('details');snapshot.className='snapshot';
      const summary=document.createElement('summary');summary.textContent='Arsip perhitungan awal NumPy seed 42 (bukan angka aktif)';snapshot.appendChild(summary);
      for(const node of Array.from(sec.children))if(!node.matches('h2,h3,.story,.source-note,[data-story]')&&!node.querySelector('[data-animation]'))snapshot.appendChild(node);
      sec.appendChild(snapshot);
    }
  }
  async function changeWeights(restore=false){
    if(!completed.has('3'))return;
    tour=false;players.forEach(p=>p.pause());
    const btn=document.getElementById('generate-weights'),rnd=document.getElementById('random-weights'),rst=document.getElementById('restore-weights'),status=document.getElementById('weight-status');
    btn.disabled=rnd.disabled=rst.disabled=true;status.textContent='Menghitung ulang bobot, forward, backward, dan 600 langkah training…';
    await new Promise(resolve=>setTimeout(resolve,30));
    try{
      const input=document.getElementById('weight-seed');
      seed=restore?42:(Number(input.value)>>>0);input.value=seed;
      generated=restore?null:E.generateParams(d.ids.length,seed);
      model=E.makeModel(d.ids,generated?E.util.cloneP(generated.P):E.util.cloneP(d.P));
      trained=E.makeModel(d.ids,E.util.cloneP(model.P));
      history=E.train(trained,[d.A,d.B],[d.mA,d.mB],d.END);refresh();revision++;embeddingReady=true;
      players.filter(p=>Number(p.key.split('.')[0])>=4).forEach(p=>{completed.delete(p.key);p.maxSeen=0;});
      archiveStatic();players.forEach(p=>{p.index=0;p.render();});
      const final=E.lossGrad(trained,[d.A,d.B],[d.mA,d.mB],d.END).L;
      const a=E.generate(trained,d.A.slice(0,pos+1),d.END).out,b=E.generate(trained,d.B.slice(0,d.B.indexOf(d.ASSIST)+1),d.END).out;
      const good=JSON.stringify(a)===JSON.stringify(d.A.slice(pos+1,-1))&&JSON.stringify(b)===JSON.stringify(d.B.slice(d.B.indexOf(d.ASSIST)+1,-1));
      const stats=document.querySelectorAll('.stats .stat b');stats[1].textContent=batch.L.toFixed(3);stats[2].textContent=final.toExponential(1);stats[3].textContent=good?'2/2':'Belum 2/2';
      for(let i=4;i<=11;i++){
        const note=document.querySelector(`#s${i} > .source-note`);
        if(!note.dataset.original)note.dataset.original=note.innerHTML;
        note.innerHTML=note.dataset.original+`<br><b>ANGKA AKTIF</b> · ${restore?'NumPy':'Mulberry32 + Box–Muller'} seed ${seed}; dihitung ulang oleh engine.js. Tabel build diarsipkan di bawah.`;
      }
      document.querySelector('footer').textContent=`Tabel arsip dihasilkan build_manual.py. Angka animasi aktif dihitung di browser (seed ${seed}); float64 JavaScript, 600 langkah AdamW.`;
      status.textContent=`Embedding & bobot seed ${seed} siap dibentuk. Tekan Putar atau Next: setiap sel diisi dari u1/u2 → z → bobot. Forward masih terkunci sampai proses ini selesai.`;
    }catch(error){status.textContent='Gagal menghitung: '+error.message;console.error(error);}finally{btn.disabled=rnd.disabled=rst.disabled=false;}
  }
  document.getElementById('generate-weights').onclick=()=>changeWeights();
  document.getElementById('random-weights').onclick=()=>{const a=new Uint32Array(1);crypto.getRandomValues(a);document.getElementById('weight-seed').value=a[0];changeWeights();};
  document.getElementById('restore-weights').onclick=()=>changeWeights(true);
  const guide=document.createElement('div');guide.className='story-guide';guide.innerHTML='<strong>Alur berurutan:</strong> mulai tahap 0 → data → token → mask → <strong>Generate embedding & bobot</strong> → forward → backward → training → jawaban. Setiap proses wajib selesai sebelum proses berikutnya terbuka. Gunakan <strong>Prev / Next</strong> untuk melihat hitungan satu per satu. Slider hanya mengulang operasi yang sudah dilihat. Jeda default di <code>&lt;html data-step-delay="850"&gt;</code> (ms). <button type="button" id="tour-play">▶ Lanjutkan tur berurutan</button> <button type="button" id="tour-stop">Ⅱ Hentikan tur</button> <span id="tour-status"></span>';
  document.querySelector('.hero').appendChild(guide);
  document.getElementById('tour-play').onclick=()=>{tour=true;players.forEach(p=>p.pause());const start=players.find(p=>!completed.has(p.key))||players[0];document.getElementById('tour-status').textContent='Tur berjalan berurutan; akan menunggu tombol Generate pada tahap embedding.';start.el.scrollIntoView({block:'center',behavior:'smooth'});start.play();};
  document.getElementById('tour-stop').onclick=()=>{tour=false;players.forEach(p=>p.pause());document.getElementById('tour-status').textContent='Tur dijeda.';};
  document.addEventListener('visibilitychange',()=>{if(document.hidden){tour=false;players.forEach(p=>p.pause());}});

  // Scroll spy memakai posisi section, termasuk section panjang dan akhir halaman.
  const sections=Array.from(document.querySelectorAll('main > section')),links=Array.from(document.querySelectorAll('nav a[href^="#s"]'));
  let pending=false,lastId='';
  function spy(){pending=false;const line=Math.min(160,window.innerHeight*.22);let current=sections[0];for(const section of sections)if(section.getBoundingClientRect().top<=line)current=section;
    if(window.scrollY+window.innerHeight>=document.documentElement.scrollHeight-4)current=sections[sections.length-1];
    if(current.id===lastId)return;lastId=current.id;
    links.forEach(a=>{const active=a.hash==='#'+current.id;a.classList.toggle('active',active);if(active)a.setAttribute('aria-current','location');else a.removeAttribute('aria-current');});
    const a=links.find(a=>a.hash==='#'+current.id),nav=document.querySelector('nav');
    if(a&&window.innerWidth>1000){const r=a.getBoundingClientRect(),n=nav.getBoundingClientRect();if(r.bottom>n.bottom||r.top<n.top)nav.scrollTop+=r.top-n.top-nav.clientHeight/2;}
  }
  const queueSpy=()=>{if(!pending){pending=true;requestAnimationFrame(spy);}};
  window.addEventListener('scroll',queueSpy,{passive:true});window.addEventListener('resize',queueSpy);window.addEventListener('hashchange',queueSpy);new ResizeObserver(queueSpy).observe(document.querySelector('main'));spy();
  // Hook read-only untuk uji browser: jalankan frame terakhir tiap proses tanpa timer.
  archiveStatic();
  updateFlow();
  window.ManualDemo={players,get seed(){return seed;},get model(){return model;},get trained(){return trained;},get revision(){return revision;},get embeddingReady(){return embeddingReady;},isUnlocked:unlocked,completed,changeWeights};
})();
