// ═══════════════════════════════════════════════════════════
// NanoLLM Transformer Visualizer — Shared Library
// Toy: d_model=32, vocab=128, seq=16, heads=4, kv_heads=2(GQA)
// Real model reference listed in REAL_CFG below
// ═══════════════════════════════════════════════════════════

// ═══ CONFIG ═══
const CFG={d:32,seq:16,V:128,nqh:4,nkvh_gqa:2,nkvh_mha:4,dh:8,ffn:128,rope_theta:10000,eps:1e-6,lr:0.001,b1:0.9,b2:0.95,wd:0.1,adam_eps:1e-8,clip:0.5,padId:15};

// ═══ REAL MODEL (transformer_100m_cuda) ═══
const REAL_CFG={d:576,seq_current:512,seq_max:2048,V:65536,nqh:9,nkvh_gqa:3,dh:64,ffn:2304,n_layers:13,rope_theta:500000,lr:0.00015,b1:0.9,b2:0.95,wd:0.1,adam_eps:1e-8,clip:0.5,padId:15,params:'101M',fp:'FP16',opt:'AdamW',weight_tying:true};

const INPUT_SENTENCES=["The quick brown fox jumps over the lazy dog.","A small bird sings well."];
const TOKENS_TEXT=["<s>","The","quick","brown","fox","jumps","over","the","lazy","dog","A","small","bird","sings","well","</s>"];
const TOKEN_IDS=[1,45,89,23,67,112,34,45,78,56,8,95,41,103,62,2];
const TARGETS=[45,89,23,67,112,34,45,78,56,8,95,41,103,62,2,0];

const BPE_WORD="quick";
const BPE_STEPS=[
    {chars:["q","u","i","c","k"],pair:null,merged:null,freq:null},
    {chars:["qu","i","c","k"],pair:["q","u"],merged:"qu",freq:87},
    {chars:["qu","ic","k"],pair:["i","c"],merged:"ic",freq:64},
    {chars:["qu","ick"],pair:["ic","k"],merged:"ick",freq:51},
    {chars:["quick"],pair:["qu","ick"],merged:"quick",freq:42}
];

const STAGE_LIST=[
    {name:'BPE Tokenization',short:'1.BPE',file:'01_bpe.html',subs:5},
    {name:'Embedding Lookup',short:'2.Embed',file:'02_embedding.html',subs:3},
    {name:'RMSNorm (Pre-Attn)',short:'3.Norm₁',file:'03_rmsnorm.html',subs:3},
    {name:'QKV Projection',short:'4.QKV',file:'04_qkv.html',subs:4},
    {name:'RoPE Encoding',short:'5.RoPE',file:'05_rope.html',subs:3},
    {name:'Attention',short:'6.Attn',file:'06_attention.html',subs:5},
    {name:'Output Proj + Residual',short:'7.Proj',file:'07_projection.html',subs:3},
    {name:'RMSNorm (Pre-FFN)',short:'8.Norm₂',file:'08_rmsnorm2.html',subs:2},
    {name:'SwiGLU FFN',short:'9.FFN',file:'09_swiglu.html',subs:5},
    {name:'Cross-Entropy Loss',short:'10.Loss',file:'10_loss.html',subs:4},
    {name:'Backpropagation',short:'11.Grad',file:'11_backprop.html',subs:5},
    {name:'AdamW Update',short:'12.Adam',file:'12_adamw.html',subs:5},
    {name:'LR Schedule',short:'13.LR',file:'13_schedule.html',subs:4},
    {name:'Curriculum',short:'14.Curr',file:'14_curriculum.html',subs:4},
    {name:'Infrastructure',short:'15.Infra',file:'15_infra.html',subs:6}
];

// ═══ PRNG ═══
function mulberry32(s){return function(){s|=0;s=s+0x6D2B79F5|0;let t=Math.imul(s^s>>>15,1|s);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296}}
const _RNG=mulberry32(42);
function R(){return+(_RNG()*2-1).toFixed(3)}
function Rg(){return+(0.8+_RNG()*0.4).toFixed(3)}
function fmt(n){if(n==null)return'?';return typeof n==='number'?n.toFixed(2):String(n)}

// ═══ MATH ═══
function matmul(A,B){
    const M=A.length,K=A[0].length,N=B[0].length;
    const C=Array.from({length:M},()=>Array(N).fill(0));
    for(let i=0;i<M;i++)for(let j=0;j<N;j++){let s=0;for(let k=0;k<K;k++)s+=A[i][k]*B[k][j];C[i][j]=+s.toFixed(4);}
    return C;
}
function transpose(M){return Array.from({length:M[0].length},(_,j)=>Array.from({length:M.length},(_,i)=>M[i][j]));}
function rmsnorm(X,gamma){
    return X.map(row=>{const m=row.reduce((a,v)=>a+v*v,0)/row.length;const rms=Math.sqrt(m+CFG.eps);return row.map((v,i)=>+((v/rms)*gamma[i]).toFixed(4));});
}
function rmsnormDetail(row,gamma){
    const sq=row.map(v=>+(v*v).toFixed(4));const sumSq=+sq.reduce((a,b)=>a+b,0).toFixed(4);
    const mean=+(sumSq/row.length).toFixed(6);const rms=+Math.sqrt(mean+CFG.eps).toFixed(4);
    const normed=row.map(v=>+(v/rms).toFixed(4));const scaled=normed.map((v,i)=>+(v*gamma[i]).toFixed(4));
    return{sq,sumSq,mean,rms,normed,scaled};
}
function softmaxRow(row){
    const mx=Math.max(...row);const exps=row.map(v=>Math.exp(v-mx));const s=exps.reduce((a,b)=>a+b,0);
    return{max:+mx.toFixed(4),shifted:row.map(v=>+(v-mx).toFixed(4)),exps:exps.map(v=>+v.toFixed(4)),sum:+s.toFixed(4),probs:exps.map(v=>+(v/s).toFixed(6))};
}
function sigmoid(x){return 1/(1+Math.exp(-x))}
function silu(x){return x*sigmoid(x)}
function applyRoPE(mat,dh,theta){
    return mat.map((row,pos)=>{const out=[...row];for(let j=0;j<row.length;j+=2){const di=j%dh;const a=pos/Math.pow(theta,di/dh);const c=Math.cos(a),s=Math.sin(a);out[j]=+(row[j]*c-row[j+1]*s).toFixed(4);out[j+1]=+(row[j]*s+row[j+1]*c).toFixed(4);}return out;});
}
function crossEntropy(probs,t){return+(-Math.log(Math.max(probs[t],1e-10))).toFixed(4)}
function getCol(M,j){return M.map(r=>r[j])}

// ═══ STATE ═══
const W={},S={};
let gqaOn=true,flashOn=false;

function initWeights(){
    const r=mulberry32(42); // reset seed
    const R2=()=>+(r()*2-1).toFixed(3);
    const Rg2=()=>+(0.8+r()*0.4).toFixed(3);
    W.embed=Array.from({length:CFG.V},()=>Array.from({length:CFG.d},()=>R2()));
    W.g1=Array.from({length:CFG.d},()=>Rg2());
    W.Wq=Array.from({length:CFG.d},()=>Array.from({length:CFG.d},()=>R2()));
    W.Wk_g=Array.from({length:CFG.d},()=>Array.from({length:CFG.nkvh_gqa*CFG.dh},()=>R2()));
    W.Wv_g=Array.from({length:CFG.d},()=>Array.from({length:CFG.nkvh_gqa*CFG.dh},()=>R2()));
    W.Wk_m=Array.from({length:CFG.d},()=>Array.from({length:CFG.nkvh_mha*CFG.dh},()=>R2()));
    W.Wv_m=Array.from({length:CFG.d},()=>Array.from({length:CFG.nkvh_mha*CFG.dh},()=>R2()));
    W.Wo=Array.from({length:CFG.d},()=>Array.from({length:CFG.d},()=>R2()));
    W.g2=Array.from({length:CFG.d},()=>Rg2());
    W.W1=Array.from({length:CFG.d},()=>Array.from({length:CFG.ffn},()=>R2()));
    W.W3=Array.from({length:CFG.d},()=>Array.from({length:CFG.ffn},()=>R2()));
    W.W2=Array.from({length:CFG.ffn},()=>Array.from({length:CFG.d},()=>R2()));
    W.gf=Array.from({length:CFG.d},()=>Rg2());
}

function forwardPass(gqa){
    gqaOn=gqa;
    const nkvh=gqa?CFG.nkvh_gqa:CFG.nkvh_mha;
    const Wk=gqa?W.Wk_g:W.Wk_m, Wv=gqa?W.Wv_g:W.Wv_m;
    S.X=TOKEN_IDS.map(id=>W.embed[id].slice());
    S.norm1=rmsnorm(S.X,W.g1); S.norm1D=S.X.map(r=>rmsnormDetail(r,W.g1));
    S.Q=matmul(S.norm1,W.Wq); S.K=matmul(S.norm1,Wk); S.V=matmul(S.norm1,Wv);
    S.Qr=applyRoPE(S.Q,CFG.dh,CFG.rope_theta); S.Kr=applyRoPE(S.K,CFG.dh,CFG.rope_theta);
    S.ropeAngles=[];
    for(let p=0;p<CFG.seq;p++){const a=[];for(let j=0;j<CFG.dh;j+=2){const ang=p/Math.pow(CFG.rope_theta,j/CFG.dh);a.push({angle:+ang.toFixed(6),cos:+Math.cos(ang).toFixed(4),sin:+Math.sin(ang).toFixed(4)});}S.ropeAngles.push(a);}
    const gqaR=CFG.nqh/nkvh; S.heads=[];
    for(let h=0;h<CFG.nqh;h++){
        const kvh=Math.floor(h/gqaR);
        const Qh=S.Qr.map(r=>r.slice(h*CFG.dh,(h+1)*CFG.dh));
        const Kh=S.Kr.map(r=>r.slice(kvh*CFG.dh,(kvh+1)*CFG.dh));
        const Vh=S.V.map(r=>r.slice(kvh*CFG.dh,(kvh+1)*CFG.dh));
        const sc=matmul(Qh,transpose(Kh));const scale=1/Math.sqrt(CFG.dh);
        const scaled=sc.map(r=>r.map(v=>+(v*scale).toFixed(4)));
        const masked=scaled.map((r,i)=>r.map((v,j)=>j>i?-1e9:v));
        const sfR=masked.map(r=>softmaxRow(r));const attnW=sfR.map(r=>r.probs);
        const out=matmul(attnW,Vh);
        S.heads.push({h,kvh,Qh,Kh,Vh,sc,scaled,masked,sfR,attnW,out});
    }
    S.concat=Array.from({length:CFG.seq},(_,i)=>{let row=[];S.heads.forEach(hd=>row.push(...hd.out[i]));return row;});
    S.proj=matmul(S.concat,W.Wo);
    S.res1=S.X.map((r,i)=>r.map((v,j)=>+(v+S.proj[i][j]).toFixed(4)));
    S.norm2=rmsnorm(S.res1,W.g2); S.norm2D=S.res1.map(r=>rmsnormDetail(r,W.g2));
    S.gate=matmul(S.norm2,W.W1); S.up=matmul(S.norm2,W.W3);
    S.siluUp=S.up.map(r=>r.map(v=>+silu(v).toFixed(4)));
    S.swiglu=S.gate.map((r,i)=>r.map((v,j)=>+(v*S.siluUp[i][j]).toFixed(4)));
    S.down=matmul(S.swiglu,W.W2);
    S.res2=S.res1.map((r,i)=>r.map((v,j)=>+(v+S.down[i][j]).toFixed(4)));
    S.normF=rmsnorm(S.res2,W.gf); S.normFD=S.res2.map(r=>rmsnormDetail(r,W.gf));
    S.logits=matmul(S.normF,transpose(W.embed));
    S.sfLog=S.logits.map(r=>softmaxRow(r));
    S.losses=TARGETS.map((t,i)=>crossEntropy(S.sfLog[i].probs,t));
    const mask=TOKEN_IDS.map(id=>id===CFG.padId?0:1);
    const maskedLoss=S.losses.map((l,i)=>l*mask[i]);
    S.mask=mask; S.maskedLoss=maskedLoss;
    S.effectiveCount=mask.reduce((a,b)=>a+b,0);
    S.totalLoss=+(maskedLoss.reduce((a,b)=>a+b,0)/S.effectiveCount).toFixed(4);
    S.dLogits=S.sfLog.map((sf,i)=>{const oh=Array(CFG.V).fill(0);oh[TARGETS[i]]=1;const denom=S.effectiveCount;return sf.probs.map((p,j)=>+(mask[i]*(p-oh[j])/denom).toFixed(6));});
    const ag=S.dLogits.flat();S.gradNorm=+Math.sqrt(ag.reduce((s,g)=>s+g*g,0)).toFixed(4);S.gradScale=S.gradNorm>CFG.clip?+(CFG.clip/S.gradNorm).toFixed(4):1;
    S.adam={param:W.Wq[0][0],grad:+(S.dLogits[0][0]*0.1).toFixed(6),m_prev:0,v_prev:0,step:1};
    const a=S.adam;a.m=+(CFG.b1*a.m_prev+(1-CFG.b1)*a.grad).toFixed(6);a.v=+(CFG.b2*a.v_prev+(1-CFG.b2)*a.grad*a.grad).toFixed(8);
    a.mHat=+(a.m/(1-Math.pow(CFG.b1,a.step))).toFixed(6);a.vHat=+(a.v/(1-Math.pow(CFG.b2,a.step))).toFixed(8);
    a.update=+(CFG.lr*a.mHat/(Math.sqrt(a.vHat)+CFG.adam_eps)).toFixed(6);a.decay=+(CFG.lr*CFG.wd*a.param).toFixed(6);
    a.newParam=+(a.param-a.update-a.decay).toFixed(4);
    S.nkvh=nkvh;
}

function initState(){initWeights();forwardPass(gqaOn);}

// ═══ RENDER HELPERS ═══
function tex(l){try{if(typeof katex!=='undefined')return katex.renderToString(l,{throwOnError:false});}catch(e){}return'<code>'+l+'</code>';}
function texBlock(l){try{if(typeof katex!=='undefined')return katex.renderToString(l,{throwOnError:false,displayMode:true});}catch(e){}return'<code>'+l+'</code>';}

// ═══ FULL-EXPLICIT RENDER — NO ABBREVIATION ═══

function htmlMat(data,id,label,opts={}){
    const R=data.length,C=data[0].length;
    const w=opts.cellW||Math.max(32,Math.min(46,Math.floor(600/Math.min(C,16))));
    let h=`<div class="mat-wrap"><div class="mat-label">${label} [${R}×${C}]</div>`;
    h+=`<div style="overflow-x:auto;overflow-y:auto;max-width:100%;max-height:320px"><div class="mat" id="${id}" style="grid-template-columns:repeat(${C},1fr)">`;
    for(let r=0;r<R;r++){
        for(let c=0;c<C;c++){
            let cls='mc';
            if(opts.hlRow===r)cls+=' hl-row';if(opts.hlCol===c)cls+=' hl-col';
            if(opts.hlCells&&opts.hlCells.some(p=>p[0]===r&&p[1]===c))cls+=' hl-cell';
            cls+=data[r][c]<0?' neg':' pos';
            h+=`<div class="${cls}" id="${id}-${r}-${c}" style="min-width:${w}px">${fmt(data[r][c])}</div>`;
        }
    }
    h+='</div></div></div>';return h;
}
function htmlHeatmap(data,id,label,opts={}){return htmlMat(data,id,label,opts);}
function htmlAuto(data,id,label,opts={}){return htmlMat(data,id,label,opts);}

function htmlVec(data,id,label,opts={}){
    const N=data.length;
    let h=`<div class="mat-wrap"><div class="mat-label">${label} [${N}]</div><div class="vec-horiz" style="overflow-x:auto;max-width:100%"><span style="color:var(--dim)">[</span>`;
    data.forEach((v,i)=>{
        let cls='vec-cell';if(opts.hlIdx===i)cls+=' hl';
        h+=`<div class="${cls}" id="${id}-${i}">${fmt(v)}</div>`;if(i<N-1)h+='<span style="color:#444">,</span>';
    });
    h+='<span style="color:var(--dim)">]</span></div></div>';return h;
}

// ─── FULL DOT PRODUCT: ALL N terms ───
function htmlFullDotProd(rowA,colB,labelR){
    const N=rowA.length;
    const prods=rowA.map((a,k)=>+(a*colB[k]).toFixed(4));
    const sum=+prods.reduce((s,p)=>s+p,0).toFixed(4);
    let h=`<div class="comp-step active"><div class="comp-line"><strong>${labelR} — dot product (${N} terms):</strong></div>`;
    h+=`<div class="comp-line" style="margin-top:3px"><strong>Perhitungan per-elemen:</strong></div>`;
    for(let k=0;k<N;k++){
        h+=`<div class="comp-line comp-terms">&nbsp;&nbsp;[${k}]: <span style="color:var(--cyan)">${fmt(rowA[k])}</span> × <span style="color:var(--purple)">${fmt(colB[k])}</span> = <span class="comp-products">${fmt(prods[k])}</span></div>`;
    }
    h+=`<div class="comp-line" style="margin-top:4px"><strong>Penjumlahan:</strong></div>`;
    h+=`<div class="comp-line comp-products">${prods.map(p=>fmt(p)).join(' + ')}</div>`;
    h+=`<div class="comp-result" style="margin-top:2px">= ${fmt(sum)}</div></div>`;
    return h;
}
// backward-compat alias
function htmlDotProd(a,b,l){return htmlFullDotProd(a,b,l);}

// ─── FULL RMSNORM: all d elements ───
function htmlFullRmsNorm(row,gamma,label){
    const d=row.length;
    const sq=row.map(v=>+(v*v).toFixed(4));
    const sumSq=+sq.reduce((a,b)=>a+b,0).toFixed(4);
    const mean=+(sumSq/d).toFixed(6);
    const rms=+Math.sqrt(mean+CFG.eps).toFixed(4);
    const normed=row.map(v=>+(v/rms).toFixed(4));
    const scaled=normed.map((v,i)=>+(v*gamma[i]).toFixed(4));
    let h=`<div class="comp-step active"><div class="comp-line"><strong>${label||'RMSNorm'} — ${d} elemen:</strong></div>`;
    h+=`<div class="comp-line" style="margin-top:3px"><strong>STEP 1: Kuadrat setiap elemen (x²):</strong></div>`;
    for(let i=0;i<d;i++)h+=`<div class="comp-line comp-terms">&nbsp;&nbsp;x[${i}]² = (${fmt(row[i])})² = ${fmt(sq[i])}</div>`;
    h+=`<div class="comp-line" style="margin-top:3px"><strong>STEP 2: Sum & Mean:</strong></div>`;
    h+=`<div class="comp-line comp-products">&nbsp;&nbsp;Σx² = ${sq.map(v=>fmt(v)).join(' + ')}</div>`;
    h+=`<div class="comp-line comp-products">&nbsp;&nbsp;Σx² = ${fmt(sumSq)}</div>`;
    h+=`<div class="comp-line comp-products">&nbsp;&nbsp;mean = ${fmt(sumSq)} / ${d} = ${fmt(mean)}</div>`;
    h+=`<div class="comp-line" style="margin-top:3px"><strong>STEP 3: RMS:</strong></div>`;
    h+=`<div class="comp-line comp-result">&nbsp;&nbsp;RMS = √(${fmt(mean)} + ${CFG.eps}) = ${fmt(rms)}</div>`;
    h+=`<div class="comp-line" style="margin-top:3px"><strong>STEP 4: Normalize (÷RMS) lalu Scale (×γ):</strong></div>`;
    for(let i=0;i<d;i++){
        h+=`<div class="comp-line comp-terms">&nbsp;&nbsp;ŷ[${i}] = (${fmt(row[i])} / ${fmt(rms)}) × γ[${i}]=${fmt(gamma[i])} = ${fmt(normed[i])} × ${fmt(gamma[i])} = <span class="comp-result">${fmt(scaled[i])}</span></div>`;
    }
    h+=`<div class="comp-line" style="margin-top:3px"><strong>Output:</strong> [${scaled.map(v=>fmt(v)).join(', ')}]</div></div>`;
    return h;
}

// ─── FULL SOFTMAX: all elements ───
function htmlFullSoftmax(row,targetIdx,label){
    const N=row.length;
    const mx=Math.max(...row);
    const shifted=row.map(v=>v<-1e8?-Infinity:+(v-mx).toFixed(4));
    const exps=shifted.map(v=>v===-Infinity?0:+Math.exp(v).toFixed(6));
    const sum=+exps.reduce((a,b)=>a+b,0).toFixed(6);
    const probs=exps.map(v=>+(v/sum).toFixed(6));
    let h=`<div class="comp-step active"><div class="comp-line"><strong>${label||'Softmax'} — ${N} elemen:</strong></div>`;
    h+=`<div class="comp-line" style="margin-top:3px"><strong>STEP 1: max = ${fmt(mx)}</strong></div>`;
    h+=`<div class="comp-line" style="margin-top:3px"><strong>STEP 2: Shifted (x - max):</strong></div>`;
    for(let i=0;i<N;i++){
        const sh=row[i]<-1e8?'−∞':fmt(+(row[i]-mx).toFixed(4));
        h+=`<div class="comp-line comp-terms">&nbsp;&nbsp;[${i}]: ${row[i]<-1e8?'−∞':fmt(row[i])} - ${fmt(mx)} = ${sh}</div>`;
    }
    h+=`<div class="comp-line" style="margin-top:3px"><strong>STEP 3: exp(shifted):</strong></div>`;
    for(let i=0;i<N;i++){
        h+=`<div class="comp-line comp-terms">&nbsp;&nbsp;exp[${i}] = e^(${shifted[i]===-Infinity?'−∞':fmt(shifted[i])}) = ${fmt(exps[i])}</div>`;
    }
    h+=`<div class="comp-line" style="margin-top:3px"><strong>STEP 4: sum = ${exps.map(v=>fmt(v)).join(' + ')}</strong></div>`;
    h+=`<div class="comp-line comp-result">&nbsp;&nbsp;sum = ${fmt(sum)}</div>`;
    h+=`<div class="comp-line" style="margin-top:3px"><strong>STEP 5: prob[i] = exp[i] / sum:</strong></div>`;
    for(let i=0;i<N;i++){
        const isT=targetIdx!==undefined&&i===targetIdx;
        h+=`<div class="comp-line${isT?' comp-result':' comp-terms'}">&nbsp;&nbsp;prob[${i}] = ${fmt(exps[i])} / ${fmt(sum)} = ${fmt(probs[i])}${isT?' ← TARGET':''}</div>`;
    }
    h+=`</div>`;return h;
}

// ─── FULL SiLU: all elements ───
function htmlFullSiLU(values,label){
    const N=values.length;
    let h=`<div class="comp-step active"><div class="comp-line"><strong>${label||'SiLU(x) = x × σ(x)'} — ${N} elemen:</strong></div>`;
    h+=`<div class="comp-line" style="margin-top:3px"><strong>Formula: SiLU(x) = x × (1/(1+e⁻ˣ))</strong></div>`;
    for(let i=0;i<N;i++){
        const x=values[i];
        const sig=+(1/(1+Math.exp(-x))).toFixed(4);
        const sil=+(x*sig).toFixed(4);
        h+=`<div class="comp-line comp-terms">&nbsp;&nbsp;[${i}]: σ(${fmt(x)}) = 1/(1+e^(${fmt(-x)})) = ${fmt(sig)} → SiLU = ${fmt(x)} × ${fmt(sig)} = <span class="comp-products">${fmt(sil)}</span></div>`;
    }
    h+=`</div>`;return h;
}

// ─── FULL RoPE: all rotation pairs ───
function htmlFullRoPE(vec,pos,dh,theta,label){
    const nPairs=Math.floor(vec.length/2);
    let h=`<div class="comp-step active"><div class="comp-line"><strong>${label||'RoPE rotation'} pos=${pos} — ${nPairs} pairs:</strong></div>`;
    h+=`<div class="comp-line"><strong>Formula: x'[2j] = x[2j]×cos(θ) - x[2j+1]×sin(θ)</strong></div>`;
    h+=`<div class="comp-line"><strong>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;x'[2j+1] = x[2j]×sin(θ) + x[2j+1]×cos(θ)</strong></div>`;
    for(let j=0;j<vec.length;j+=2){
        const di=j%(dh||vec.length);
        const angle=pos/Math.pow(theta||CFG.rope_theta,di/(dh||vec.length));
        const cos_a=+Math.cos(angle).toFixed(4),sin_a=+Math.sin(angle).toFixed(4);
        const x0=vec[j],x1=vec[j+1];
        const r0=+(x0*cos_a-x1*sin_a).toFixed(4);
        const r1=+(x0*sin_a+x1*cos_a).toFixed(4);
        h+=`<div class="comp-line" style="margin-top:3px;border-left:2px solid var(--cyan);padding-left:6px"><strong>Pair ${j/2} (dims ${j},${j+1}):</strong></div>`;
        h+=`<div class="comp-line comp-terms">&nbsp;&nbsp;θ = ${pos} / (${theta||CFG.rope_theta}^(${di}/${dh||vec.length})) = ${fmt(angle)}</div>`;
        h+=`<div class="comp-line comp-terms">&nbsp;&nbsp;cos(${fmt(angle)}) = ${fmt(cos_a)}, sin(${fmt(angle)}) = ${fmt(sin_a)}</div>`;
        h+=`<div class="comp-line comp-terms">&nbsp;&nbsp;x'[${j}] = ${fmt(x0)}×${fmt(cos_a)} - ${fmt(x1)}×${fmt(sin_a)} = ${fmt(+(x0*cos_a).toFixed(4))} - ${fmt(+(x1*sin_a).toFixed(4))} = <span class="comp-result">${fmt(r0)}</span></div>`;
        h+=`<div class="comp-line comp-terms">&nbsp;&nbsp;x'[${j+1}] = ${fmt(x0)}×${fmt(sin_a)} + ${fmt(x1)}×${fmt(cos_a)} = ${fmt(+(x0*sin_a).toFixed(4))} + ${fmt(+(x1*cos_a).toFixed(4))} = <span class="comp-result">${fmt(r1)}</span></div>`;
    }
    h+=`</div>`;return h;
}

// ─── FULL Residual: all elements ───
function htmlFullResidual(a,b,label){
    const N=a.length;
    const result=a.map((v,i)=>+(v+b[i]).toFixed(4));
    let h=`<div class="comp-step active"><div class="comp-line"><strong>${label||'Residual'} — ${N} elemen (a + b = result):</strong></div>`;
    for(let i=0;i<N;i++){
        h+=`<div class="comp-line comp-terms">&nbsp;&nbsp;[${i}]: ${fmt(a[i])} + ${fmt(b[i])} = <span class="comp-result">${fmt(result[i])}</span></div>`;
    }
    h+=`</div>`;return h;
}

// ─── FULL Element-wise multiply ───
function htmlFullElemMul(a,b,label){
    const N=a.length;
    let h=`<div class="comp-step active"><div class="comp-line"><strong>${label||'Element-wise ⊙'} — ${N} elemen:</strong></div>`;
    for(let i=0;i<N;i++){
        const p=+(a[i]*b[i]).toFixed(4);
        h+=`<div class="comp-line comp-terms">&nbsp;&nbsp;[${i}]: ${fmt(a[i])} × ${fmt(b[i])} = <span class="comp-products">${fmt(p)}</span></div>`;
    }
    h+=`</div>`;return h;
}

// ─── FULL Scale (multiply all by constant) ───
function htmlFullScale(values,factor,label){
    const N=values.length;
    let h=`<div class="comp-step active"><div class="comp-line"><strong>${label||'Scale'} — ×${fmt(factor)}, ${N} elemen:</strong></div>`;
    for(let i=0;i<N;i++){
        const r=+(values[i]*factor).toFixed(4);
        h+=`<div class="comp-line comp-terms">&nbsp;&nbsp;[${i}]: ${fmt(values[i])} × ${fmt(factor)} = <span class="comp-products">${fmt(r)}</span></div>`;
    }
    h+=`</div>`;return h;
}

function onCellClick(id,r,c){
    const cd=document.getElementById('comp-detail');
    if(cd)cd.innerHTML=`<div class="comp-step active"><div class="comp-line">Cell [${r},${c}] selected</div></div>`;
}

function htmlTokenSelector(selectedIdx,onChange){
    let h='<div class="selector">';
    const show=TOKENS_TEXT.length>10?TOKENS_TEXT.slice(0,10):TOKENS_TEXT;
    show.forEach((t,i)=>{h+=`<div class="sel-btn${i===selectedIdx?' active':''}" onclick="${onChange}(${i})">${i}:${t.length>5?t.slice(0,4)+'…':t}</div>`;});
    if(TOKENS_TEXT.length>10)h+=`<div style="color:var(--dim);font-size:.6rem;padding:3px">+${TOKENS_TEXT.length-10} more</div>`;
    h+='</div>';return h;
}

function htmlHeadSelector(n,selectedIdx,onChange){
    let h='<div class="selector">';
    for(let i=0;i<n;i++)h+=`<div class="sel-btn${i===selectedIdx?' active':''}" onclick="${onChange}(${i})">H${i}</div>`;
    h+='</div>';return h;
}

function htmlViewToggle(){return '';}
function setView(m){}

// ═══ ANIMATION ═══
let animId=0,animSpeed=5;
function cancelAnim(){animId++}
function getDelay(){return Math.max(50,1100-animSpeed*100)}
function aDelay(ms){const my=animId;return new Promise(r=>{setTimeout(()=>r(animId===my),ms||getDelay())});}

async function runMatmulAnim(idA,idB,idR,A,B,R_,maxCells){
    const my=++animId;const rows=R_.length,cols=R_[0].length;
    const limit=maxCells||Math.min(rows*cols,32);let count=0;
    for(let i=0;i<rows&&count<limit;i++){for(let j=0;j<cols&&count<limit;j++){
        if(animId!==my)return;count++;
        document.querySelectorAll(`#${idA} .mc,#${idA} .hc`).forEach(el=>el.style.outline='');
        document.querySelectorAll(`#${idB} .mc,#${idB} .hc`).forEach(el=>el.style.outline='');
        for(let k=0;k<A[0].length;k++){const c=document.getElementById(`${idA}-${i}-${k}`);if(c)c.style.outline='1px solid var(--yellow)';}
        for(let k=0;k<B.length;k++){const c=document.getElementById(`${idB}-${k}-${j}`);if(c)c.style.outline='1px solid var(--purple)';}
        const cr=document.getElementById(`${idR}-${i}-${j}`);
        if(cr){cr.style.outline='2px solid var(--green)';cr.textContent=fmt(R_[i][j]);}
        const cd=document.getElementById('comp-detail');
        if(cd)cd.innerHTML=htmlDotProd(A[i],getCol(B,j),`[${i}][${j}]`);
        if(!await aDelay())return;
    }}
}

async function animGradFlow(n){
    const my=++animId;
    for(let i=0;i<n;i++){if(animId!==my)return;const el=document.getElementById(`gb-${i}`);if(el)el.classList.add('active');if(!await aDelay(getDelay()*0.5))return;}
}

// ═══ PAGE FRAMEWORK ═══
const PAGE={stage:0,sub:0,maxSub:1,render:null};

function initStagePage(stageIdx,maxSubs,renderFn){
    PAGE.stage=stageIdx;PAGE.maxSub=maxSubs;PAGE.render=renderFn;PAGE.sub=0;
    window.addEventListener('message',function(e){
        if(!e.data)return;
        if(e.data.type==='settings'){
            animSpeed=e.data.speed||5;
            const newGqa=e.data.gqa!==undefined?e.data.gqa:gqaOn;
            flashOn=e.data.flash||false;
            if(newGqa!==gqaOn){gqaOn=newGqa;forwardPass(gqaOn);}
            renderFn(PAGE.sub);
        }
    });
    document.addEventListener('keydown',function(e){
        if(e.key==='ArrowRight'||e.key===' '){e.preventDefault();navSub(1);}
        if(e.key==='ArrowLeft'){e.preventDefault();navSub(-1);}
    });
    try{parent.postMessage({type:'stageReady',stage:stageIdx},'*');}catch(e){}
    updateSubDots();renderFn(0);
}

function navSub(dir){
    cancelAnim();
    if(dir>0){
        if(PAGE.sub<PAGE.maxSub-1){PAGE.sub++;PAGE.render(PAGE.sub);updateSubDots();}
        else{try{parent.postMessage({type:'requestNav',dir:1},'*');}catch(e){}}
    }else{
        if(PAGE.sub>0){PAGE.sub--;PAGE.render(PAGE.sub);updateSubDots();}
        else{try{parent.postMessage({type:'requestNav',dir:-1},'*');}catch(e){}}
    }
}
function jumpSub(i){cancelAnim();PAGE.sub=i;PAGE.render(PAGE.sub);updateSubDots();}

function updateSubDots(){
    const el=document.getElementById('substep-dots');if(!el)return;
    el.innerHTML=Array.from({length:PAGE.maxSub},(_,i)=>
        `<div class="sdot${i===PAGE.sub?' active':i<PAGE.sub?' done':''}" onclick="jumpSub(${i})"></div>`
    ).join('');
    const info=document.getElementById('substep-info');
    if(info)info.textContent=`Sub-step ${PAGE.sub+1}/${PAGE.maxSub}`;
}
