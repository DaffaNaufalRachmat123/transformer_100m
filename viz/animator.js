// ═══════════════════════════════════════════════════
// NanoLLM Sequential Animator Engine
// Tables start EMPTY → fill cell-by-cell with math
// ═══════════════════════════════════════════════════

const STEPS=[],SMETA=[];
let cur=0,playing=false,ptimer=null,spd=60;

function addStage(name){SMETA.push({name,start:STEPS.length});}
function addStep(label,mathFn,actionFn){STEPS.push({si:SMETA.length-1,label,mf:mathFn,af:actionFn});}

// ── Grid helpers ──
function makeGrid(id,R,C,label,data){
    const w=Math.max(26,Math.min(40,Math.floor(480/Math.min(C,16))));
    let h=`<div class="ag-wrap"><div class="ag-label">${label} [${R}×${C}]</div><div class="ag-scroll"><div class="mat" id="${id}" style="grid-template-columns:repeat(${C},1fr)">`;
    for(let r=0;r<R;r++)for(let c=0;c<C;c++){
        if(data){const v=data[r][c];h+=`<div class="mc${v<0?' neg':' pos'}" id="${id}-${r}-${c}" style="min-width:${w}px">${fmt(v)}</div>`;}
        else h+=`<div class="mc empty" id="${id}-${r}-${c}" style="min-width:${w}px">·</div>`;
    }
    h+='</div></div></div>';return h;
}
function fc(id,r,c,v){const e=document.getElementById(`${id}-${r}-${c}`);if(!e)return;e.textContent=fmt(v);e.className='mc hl-fill'+(v<0?' neg':' pos');}
function clr(){document.querySelectorAll('.hl-src').forEach(e=>e.classList.remove('hl-src'));}
function hlR(id,r,C){for(let c=0;c<C;c++){const e=document.getElementById(`${id}-${r}-${c}`);if(e)e.classList.add('hl-src');}}
function hlC(id,c,R){for(let r=0;r<R;r++){const e=document.getElementById(`${id}-${r}-${c}`);if(e)e.classList.add('hl-src');}}

// ── Math HTML helper for dot products ──
function mDot(rA,cB,label,nA,ri,nB,ci){
    const N=rA.length,pr=rA.map((a,k)=>+(a*cB[k]).toFixed(4));
    const s=+pr.reduce((a,b)=>a+b,0).toFixed(4);
    let h=`<b>${label}</b> = Σ ${nA}[${ri}][k]×${nB}[k][${ci}] (k=0..${N-1})<br><br>`;
    for(let k=0;k<N;k++)h+=`[${k}]: ${fmt(rA[k])} × ${fmt(cB[k])} = <span class="cp">${fmt(pr[k])}</span><br>`;
    h+=`<br><b>Σ = <span class="cr">${fmt(s)}</span></b>`;return h;
}

// ── Generic matmul step builder ──
function addMM(A,B,res,oId,nA,nB,nO,R,K,C,iA,iB){
    for(let r=0;r<R;r++)for(let c=0;c<C;c++){
        const rw=A[r],cl=getCol(B,c),v=res[r][c];
        addStep(`${nO}[${r}][${c}]=${fmt(v)}`,()=>mDot(rw,cl,`${nO}[${r}][${c}]`,nA,r,nB,c),
            ()=>{clr();if(iA)hlR(iA,r,K);if(iB)hlC(iB,c,K);fc(oId,r,c,v);});
    }
}

// ═══════════════════════════════════════
// STEP BUILDERS — all 12 stages
// ═══════════════════════════════════════
function buildAll(){
    // Stage 0: BPE
    addStage('1. BPE Tokenization');
    BPE_STEPS.forEach((st,i)=>{
        if(!i)addStep('Karakter awal',()=>`Kata "${BPE_WORD}" → [${st.chars.join(', ')}]`,()=>{el('bpe-out').innerHTML=st.chars.map(c=>`<span class="tb">${c}</span>`).join(' ');});
        else addStep(`Merge→${st.merged}`,()=>`Merge "${st.pair[0]}"+"${st.pair[1]}" (freq=${st.freq}) → "${st.merged}"<br>Hasil: [${st.chars.join(', ')}]`,()=>{el('bpe-out').innerHTML=st.chars.map(c=>`<span class="tb${c===st.merged?' hl':''}">${c}</span>`).join(' ');});
    });
    TOKENS_TEXT.forEach((t,i)=>{
        addStep(`Token${i}→${TOKEN_IDS[i]}`,()=>`"${t}" → vocab lookup → ID ${TOKEN_IDS[i]}`,()=>{const e=el(`tid-${i}`);if(e){e.textContent=`${t}[${TOKEN_IDS[i]}]`;e.className='tb hl';}});
    });

    // Stage 1: Embedding
    addStage('2. Embedding Lookup');
    for(let t=0;t<CFG.seq;t++)for(let d=0;d<CFG.d;d++){
        const id=TOKEN_IDS[t],v=W.embed[id][d];
        addStep(`X[${t}][${d}]=${fmt(v)}`,()=>`<b>X[${t}][${d}]</b> = W_embed[${id}][${d}]<br>Token "${TOKENS_TEXT[t]}" → ID ${id}, kolom ${d}<br><br><span class="cr">= ${fmt(v)}</span>`,
            ()=>{clr();hlR('We',id,CFG.d);fc('X',t,d,v);});
    }

    // Stage 2: RMSNorm Pre-Attn
    addStage('3. RMSNorm (Pre-Attn)');
    for(let t=0;t<CFG.seq;t++){
        const det=S.norm1D[t],row=S.X[t];
        addStep(`RMS[${t}]=${fmt(det.rms)}`,()=>{let h=`<b>RMSNorm token ${t}:</b><br><br><b>x²:</b><br>`;
            row.forEach((v,i)=>h+=`[${i}]: (${fmt(v)})²=${fmt(det.sq[i])}<br>`);
            h+=`<br>Σx²=${fmt(det.sumSq)}, mean=${fmt(det.mean)}<br><b>RMS=√(${fmt(det.mean)}+ε)=<span class="cr">${fmt(det.rms)}</span></b>`;return h;},()=>{});
        for(let d=0;d<CFG.d;d++){
            addStep(`X̂[${t}][${d}]=${fmt(det.scaled[d])}`,()=>`ŷ[${t}][${d}]=(${fmt(row[d])}/${fmt(det.rms)})×γ[${d}]=${fmt(W.g1[d])}<br>=${fmt(det.normed[d])}×${fmt(W.g1[d])}<br><span class="cr">=${fmt(det.scaled[d])}</span>`,
                ()=>fc('Xn',t,d,det.scaled[d]));
        }
    }

    // Stage 3: QKV Projection
    addStage('4. QKV Projection');
    const nkv=gqaOn?CFG.nkvh_gqa:CFG.nkvh_mha,kvD=nkv*CFG.dh;
    const Wk=gqaOn?W.Wk_g:W.Wk_m,Wv=gqaOn?W.Wv_g:W.Wv_m;
    addMM(S.norm1,W.Wq,S.Q,'Q','X̂','W_Q','Q',CFG.seq,CFG.d,CFG.d,'Xn2','Wq');
    addMM(S.norm1,Wk,S.K,'K','X̂','W_K','K',CFG.seq,CFG.d,kvD,'Xn2','Wk');
    addMM(S.norm1,Wv,S.V,'V','X̂','W_V','V',CFG.seq,CFG.d,kvD,'Xn2','Wv');

    // Stage 4: RoPE
    addStage('5. RoPE Encoding');
    [['Q',S.Q,S.Qr,'Qr'],['K',S.K,S.Kr,'Kr']].forEach(([nm,src,dst,oid])=>{
        for(let t=0;t<CFG.seq;t++)for(let j=0;j<src[0].length;j+=2){
            const di=j%CFG.dh,ang=t/Math.pow(CFG.rope_theta,di/CFG.dh);
            const co=+Math.cos(ang).toFixed(4),si=+Math.sin(ang).toFixed(4);
            const x0=src[t][j],x1=src[t][j+1],r0=dst[t][j],r1=dst[t][j+1];
            addStep(`${nm}'[${t}][${j}]`,
                ()=>`<b>RoPE ${nm}[${t}] dims(${j},${j+1}):</b><br>θ=${fmt(ang)}, cos=${fmt(co)}, sin=${fmt(si)}<br><br>${nm}'[${j}]=${fmt(x0)}×${fmt(co)}−${fmt(x1)}×${fmt(si)}=${fmt(+(x0*co).toFixed(4))}−${fmt(+(x1*si).toFixed(4))}<br><span class="cr">=${fmt(r0)}</span><br><br>${nm}'[${j+1}]=${fmt(x0)}×${fmt(si)}+${fmt(x1)}×${fmt(co)}=${fmt(+(x0*si).toFixed(4))}+${fmt(+(x1*co).toFixed(4))}<br><span class="cr">=${fmt(r1)}</span>`,
                ()=>{fc(oid,t,j,r0);fc(oid,t,j+1,r1);});
        }
    });

    // Stage 5: Attention (4 heads)
    addStage('6. Attention (4 heads)');
    for(let h=0;h<CFG.nqh;h++){
        const hd=S.heads[h],KT=transpose(hd.Kh),sc=1/Math.sqrt(CFG.dh);
        // Score
        for(let r=0;r<CFG.seq;r++)for(let c=0;c<CFG.seq;c++){
            const pr=hd.Qh[r].map((a,k)=>+(a*KT[k][c]).toFixed(4)),sm=+pr.reduce((a,b)=>a+b,0).toFixed(4);
            addStep(`Sc${h}[${r}][${c}]`,()=>{let m=`<b>Score H${h}[${r}][${c}]</b>=Q·Kᵀ<br><br>`;pr.forEach((p,k)=>m+=`[${k}]:${fmt(hd.Qh[r][k])}×${fmt(KT[k][c])}=${fmt(p)}<br>`);m+=`<br><span class="cr">Σ=${fmt(sm)}</span>`;return m;},
                ()=>fc(`sc${h}`,r,c,sm));
        }
        // Scale
        for(let r=0;r<CFG.seq;r++)for(let c=0;c<CFG.seq;c++){
            const v=hd.scaled[r][c];
            addStep(`Scl${h}[${r}][${c}]`,()=>`Scaled=${fmt(hd.sc[r][c])}×${fmt(sc)}<br><span class="cr">=${fmt(v)}</span>`,()=>fc(`scl${h}`,r,c,v));
        }
        // Mask
        for(let r=0;r<CFG.seq;r++)for(let c=0;c<CFG.seq;c++){
            const mk=c>r;
            addStep(`Msk${h}[${r}][${c}]`,()=>mk?`j=${c}>i=${r} → <span class="cr">−∞</span> (causal)`:`j=${c}≤i=${r} → keep <span class="cr">${fmt(hd.scaled[r][c])}</span>`,
                ()=>{const e=document.getElementById(`msk${h}-${r}-${c}`);if(e){e.textContent=mk?'-∞':fmt(hd.scaled[r][c]);e.className='mc hl-fill'+(mk?' neg':' pos');}});
        }
        // Softmax
        for(let r=0;r<CFG.seq;r++){
            const sf=hd.sfR[r];
            for(let c=0;c<CFG.seq;c++){
                const p=sf.probs[c],mv=hd.masked[r][c];
                addStep(`Attn${h}[${r}][${c}]`,()=>{const sh=mv<-1e8?'−∞':fmt(+(mv-sf.max).toFixed(4)),ex=mv<-1e8?'0':fmt(+Math.exp(mv-sf.max).toFixed(6));
                    return`<b>Softmax H${h} [${r}][${c}]:</b><br>max=${fmt(sf.max)}<br>shifted=${sh}<br>exp=${ex}<br>sum=${fmt(sf.sum)}<br><span class="cr">prob=${ex}/${fmt(sf.sum)}=${fmt(p)}</span>`;},
                    ()=>fc(`atn${h}`,r,c,p));
            }
        }
        // Output = Attn × V
        for(let r=0;r<CFG.seq;r++)for(let c=0;c<CFG.dh;c++){
            const rw=hd.attnW[r],cl=getCol(hd.Vh,c),v=hd.out[r][c];
            addStep(`O${h}[${r}][${c}]`,()=>mDot(rw,cl,`Out_H${h}[${r}][${c}]`,'Attn',r,'V',c),()=>fc(`oh${h}`,r,c,v));
        }
    }

    // Stage 6: Proj + Residual
    addStage('7. Output Proj + Residual');
    for(let t=0;t<CFG.seq;t++)for(let d=0;d<CFG.d;d++){
        const hi=Math.floor(d/CFG.dh),di=d%CFG.dh;
        addStep(`Cat[${t}][${d}]`,()=>`Concat[${t}][${d}]=Head${hi}_out[${t}][${di}]<br><span class="cr">=${fmt(S.concat[t][d])}</span>`,()=>fc('cat',t,d,S.concat[t][d]));
    }
    addMM(S.concat,W.Wo,S.proj,'prj','Cat','W_O','Proj',CFG.seq,CFG.d,CFG.d,'cat','Wo');
    for(let t=0;t<CFG.seq;t++)for(let d=0;d<CFG.d;d++){
        addStep(`Xm[${t}][${d}]`,()=>`X_mid=${fmt(S.X[t][d])}+${fmt(S.proj[t][d])}<br><span class="cr">=${fmt(S.res1[t][d])}</span>`,()=>fc('rs1',t,d,S.res1[t][d]));
    }

    // Stage 7: RMSNorm Pre-FFN
    addStage('8. RMSNorm (Pre-FFN)');
    for(let t=0;t<CFG.seq;t++){
        const det=S.norm2D[t],row=S.res1[t];
        addStep(`RMS2[${t}]`,()=>{let h=`<b>RMSNorm token ${t}:</b><br>`;row.forEach((v,i)=>h+=`[${i}]:(${fmt(v)})²=${fmt(det.sq[i])}<br>`);h+=`Σ=${fmt(det.sumSq)},mean=${fmt(det.mean)}<br><span class="cr">RMS=${fmt(det.rms)}</span>`;return h;},()=>{});
        for(let d=0;d<CFG.d;d++)addStep(`X̂₂[${t}][${d}]`,()=>`(${fmt(row[d])}/${fmt(det.rms)})×${fmt(W.g2[d])}<br><span class="cr">=${fmt(det.scaled[d])}</span>`,()=>fc('n2',t,d,det.scaled[d]));
    }

    // Stage 8: SwiGLU FFN
    addStage('9. SwiGLU FFN');
    addMM(S.norm2,W.W1,S.gate,'gt','X̂₂','W₁','Gate',CFG.seq,CFG.d,CFG.ffn,null,null);
    addMM(S.norm2,W.W3,S.up,'up','X̂₂','W₃','Up',CFG.seq,CFG.d,CFG.ffn,null,null);
    for(let t=0;t<CFG.seq;t++)for(let d=0;d<CFG.ffn;d++){
        const x=S.up[t][d],sg=+(1/(1+Math.exp(-x))).toFixed(4),v=S.siluUp[t][d];
        addStep(`SiLU[${t}][${d}]`,()=>`σ(${fmt(x)})=1/(1+e^${fmt(-x)})=${fmt(sg)}<br>SiLU=${fmt(x)}×${fmt(sg)}<br><span class="cr">=${fmt(v)}</span>`,()=>fc('slu',t,d,v));
    }
    for(let t=0;t<CFG.seq;t++)for(let d=0;d<CFG.ffn;d++)
        addStep(`Swi[${t}][${d}]`,()=>`${fmt(S.gate[t][d])}×${fmt(S.siluUp[t][d])}<br><span class="cr">=${fmt(S.swiglu[t][d])}</span>`,()=>fc('sw',t,d,S.swiglu[t][d]));
    addMM(S.swiglu,W.W2,S.down,'dn','SwiGLU','W₂','Down',CFG.seq,CFG.ffn,CFG.d,null,null);
    for(let t=0;t<CFG.seq;t++)for(let d=0;d<CFG.d;d++)
        addStep(`Xo[${t}][${d}]`,()=>`${fmt(S.res1[t][d])}+${fmt(S.down[t][d])}<br><span class="cr">=${fmt(S.res2[t][d])}</span>`,()=>fc('rs2',t,d,S.res2[t][d]));

    // Stage 9: Loss
    addStage('10. Cross-Entropy Loss');
    // Final RMSNorm
    for(let t=0;t<CFG.seq;t++){
        const det=S.normFD[t],row=S.res2[t];
        addStep(`RMSf[${t}]`,()=>{let h=`<b>Final RMSNorm ${t}:</b><br>`;row.forEach((v,i)=>h+=`[${i}]:(${fmt(v)})²=${fmt(det.sq[i])}<br>`);h+=`<span class="cr">RMS=${fmt(det.rms)}</span>`;return h;},()=>{});
        for(let d=0;d<CFG.d;d++)addStep(`X̂f[${t}][${d}]`,()=>`(${fmt(row[d])}/${fmt(det.rms)})×${fmt(W.gf[d])}<br><span class="cr">=${fmt(det.scaled[d])}</span>`,()=>fc('nf',t,d,det.scaled[d]));
    }
    const WT=transpose(W.embed);
    addMM(S.normF,WT,S.logits,'lg','X̂f','Wₑᵀ','Logits',CFG.seq,CFG.d,CFG.V,null,null);
    for(let t=0;t<CFG.seq;t++){
        const sf=S.sfLog[t],tg=TARGETS[t];
        for(let v=0;v<CFG.V;v++){
            const p=sf.probs[v];
            addStep(`P[${t}][${v}]`,()=>{const sh=fmt(+(S.logits[t][v]-sf.max).toFixed(4)),ex=fmt(+Math.exp(Math.max(-20,S.logits[t][v]-sf.max)).toFixed(6));
                return`exp(${fmt(S.logits[t][v])}−${fmt(sf.max)})/${fmt(sf.sum)}<br>=exp(${sh})/${fmt(sf.sum)}=${ex}/${fmt(sf.sum)}<br><span class="cr">=${fmt(p)}</span>${v===tg?' ← TARGET':''}`;},()=>fc('pb',t,v,p));
        }
        addStep(`Loss[${t}]=${fmt(S.losses[t])}`,()=>`−log(P[${tg}])=−log(${fmt(sf.probs[tg])})<br><span class="cr">=${fmt(S.losses[t])}</span>`,
            ()=>{const e=document.getElementById(`ls-${t}`);if(e){e.textContent=fmt(S.losses[t]);e.className='mc hl-fill neg';}});
    }
    addStep(`Loss=${fmt(S.totalLoss)}`,()=>`Mean=(${S.losses.map(v=>fmt(v)).join('+')})/16<br><span class="cr">=${fmt(S.totalLoss)}</span>`,
        ()=>{const e=el('tot-loss');if(e){e.textContent='Loss='+fmt(S.totalLoss);e.className='cr';}});

    // Stage 10: Backprop
    addStage('11. Backpropagation');
    for(let t=0;t<CFG.seq;t++){const sf=S.sfLog[t],tg=TARGETS[t];
        for(let v=0;v<CFG.V;v++){const isT=v===tg?1:0;
            addStep(`dL[${t}][${v}]`,()=>`(${fmt(sf.probs[v])}−${isT})/${CFG.seq}<br><span class="cr">=${S.dLogits[t][v]}</span>${isT?' ←TGT':''}`,()=>fc('dl',t,v,S.dLogits[t][v]));
        }
    }
    addStep(`‖∇‖=${fmt(S.gradNorm)}`,()=>`√(Σgᵢ²)=<span class="cr">${fmt(S.gradNorm)}</span><br>${S.gradNorm>1?'Clip: scale='+fmt(S.gradScale):'No clip'}`,()=>{});

    // Stage 11: AdamW
    addStage('12. AdamW Update');
    const a=S.adam;
    addStep('m_t',()=>`m=${CFG.b1}×${fmt(a.m_prev)}+(1−${CFG.b1})×${a.grad}<br><span class="cr">=${a.m}</span>`,()=>{});
    addStep('v_t',()=>`v=${CFG.b2}×${fmt(a.v_prev)}+(1−${CFG.b2})×${a.grad}²<br><span class="cr">=${a.v}</span>`,()=>{});
    addStep('m̂',()=>`m̂=${a.m}/(1−${CFG.b1}^${a.step})<br><span class="cr">=${a.mHat}</span>`,()=>{});
    addStep('v̂',()=>`v̂=${a.v}/(1−${CFG.b2}^${a.step})<br><span class="cr">=${a.vHat}</span>`,()=>{});
    addStep('update',()=>`${CFG.lr}×${a.mHat}/(√${a.vHat}+ε)<br><span class="cr">=${a.update}</span>`,()=>{});
    addStep('decay',()=>`${CFG.lr}×${CFG.wd}×${fmt(a.param)}<br><span class="cr">=${a.decay}</span>`,()=>{});
    addStep('θ_new',()=>`θ=${fmt(a.param)}−${a.update}−${a.decay}<br><span class="cr">=${fmt(a.newParam)}</span><br><br><b>✓ Training step selesai! Loss=${fmt(S.totalLoss)}</b>`,()=>{});
}

// ═══════════════════════════════════════
// LAYOUT BUILDERS — rebuild matrix panel per stage
// ═══════════════════════════════════════
function el(id){return document.getElementById(id);}
const LAYOUTS=[
    // 0: BPE
    ()=>`<div class="ag-sec"><h4>BPE Merge Demo: "${BPE_WORD}"</h4><div id="bpe-out" class="bpe-area"></div>
        <h4>Token IDs [${CFG.seq}]:</h4><div class="tok-row">${TOKENS_TEXT.map((_,i)=>`<span class="tb" id="tid-${i}">·</span>`).join(' ')}</div></div>`,
    // 1: Embedding
    ()=>`<div class="ag-sec"><h4>W_embed [${CFG.V}×${CFG.d}] (bobot)</h4>${makeGrid('We',CFG.V,CFG.d,'',W.embed)}</div>
        <div class="ag-sec"><h4>X [${CFG.seq}×${CFG.d}] — OUTPUT (diisi)</h4>${makeGrid('X',CFG.seq,CFG.d,'')}</div>`,
    // 2: RMSNorm1
    ()=>`<div class="ag-sec"><h4>Input X [${CFG.seq}×${CFG.d}] (dari Embedding)</h4>${makeGrid('Xin',CFG.seq,CFG.d,'',S.X)}</div>
        <div class="ag-sec"><h4>γ [${CFG.d}]</h4>${makeGrid('g1',1,CFG.d,'',[ W.g1])}</div>
        <div class="ag-sec"><h4>X̂ [${CFG.seq}×${CFG.d}] — OUTPUT</h4>${makeGrid('Xn',CFG.seq,CFG.d,'')}</div>`,
    // 3: QKV
    ()=>{const kd=gqaOn?CFG.nkvh_gqa*CFG.dh:CFG.d;
        return`<div class="ag-sec"><h4>X̂ (input)</h4>${makeGrid('Xn2',CFG.seq,CFG.d,'',S.norm1)}</div>
        <div class="ag-sec"><h4>W_Q [${CFG.d}×${CFG.d}]</h4>${makeGrid('Wq',CFG.d,CFG.d,'',W.Wq)}</div>
        <div class="ag-sec"><h4>Q — OUTPUT</h4>${makeGrid('Q',CFG.seq,CFG.d,'')}</div>
        <div class="ag-sec"><h4>W_K [${CFG.d}×${kd}]</h4>${makeGrid('Wk',CFG.d,kd,'',(gqaOn?W.Wk_g:W.Wk_m))}</div>
        <div class="ag-sec"><h4>K — OUTPUT</h4>${makeGrid('K',CFG.seq,kd,'')}</div>
        <div class="ag-sec"><h4>W_V [${CFG.d}×${kd}]</h4>${makeGrid('Wv',CFG.d,kd,'',(gqaOn?W.Wv_g:W.Wv_m))}</div>
        <div class="ag-sec"><h4>V — OUTPUT</h4>${makeGrid('V',CFG.seq,kd,'')}</div>`;},
    // 4: RoPE
    ()=>`<div class="ag-sec"><h4>Q (sebelum RoPE)</h4>${makeGrid('Qb',CFG.seq,CFG.d,'',S.Q)}</div>
        <div class="ag-sec"><h4>Q' — OUTPUT (setelah RoPE)</h4>${makeGrid('Qr',CFG.seq,CFG.d,'')}</div>
        <div class="ag-sec"><h4>K' — OUTPUT</h4>${makeGrid('Kr',CFG.seq,S.K[0].length,'')}</div>`,
    // 5: Attention
    ()=>{let h='';for(let i=0;i<CFG.nqh;i++){
        h+=`<div class="ag-sec"><h4>── Head ${i} (KV head ${S.heads[i].kvh}) ──</h4>
            <div class="ag-row">${makeGrid(`sc${i}`,CFG.seq,CFG.seq,'Score')} ${makeGrid(`scl${i}`,CFG.seq,CFG.seq,'Scaled')} ${makeGrid(`msk${i}`,CFG.seq,CFG.seq,'Masked')}</div>
            <div class="ag-row">${makeGrid(`atn${i}`,CFG.seq,CFG.seq,'Attn Weights')} ${makeGrid(`oh${i}`,CFG.seq,CFG.dh,'Output')}</div></div>`;}return h;},
    // 6: Proj+Residual
    ()=>`<div class="ag-sec"><h4>Concat — OUTPUT</h4>${makeGrid('cat',CFG.seq,CFG.d,'')}</div>
        <div class="ag-sec"><h4>W_O [${CFG.d}×${CFG.d}]</h4>${makeGrid('Wo',CFG.d,CFG.d,'',W.Wo)}</div>
        <div class="ag-sec"><h4>Proj — OUTPUT</h4>${makeGrid('prj',CFG.seq,CFG.d,'')}</div>
        <div class="ag-sec"><h4>X_mid (X+Proj) — OUTPUT</h4>${makeGrid('rs1',CFG.seq,CFG.d,'')}</div>`,
    // 7: RMSNorm2
    ()=>`<div class="ag-sec"><h4>X_mid (input)</h4>${makeGrid('r1b',CFG.seq,CFG.d,'',S.res1)}</div>
        <div class="ag-sec"><h4>X̂₂ — OUTPUT</h4>${makeGrid('n2',CFG.seq,CFG.d,'')}</div>`,
    // 8: SwiGLU
    ()=>`<div class="ag-sec"><h4>Gate — OUTPUT [${CFG.seq}×${CFG.ffn}]</h4>${makeGrid('gt',CFG.seq,CFG.ffn,'')}</div>
        <div class="ag-sec"><h4>Up — OUTPUT</h4>${makeGrid('up',CFG.seq,CFG.ffn,'')}</div>
        <div class="ag-sec"><h4>SiLU(Up) — OUTPUT</h4>${makeGrid('slu',CFG.seq,CFG.ffn,'')}</div>
        <div class="ag-sec"><h4>SwiGLU (Gate⊙SiLU) — OUTPUT</h4>${makeGrid('sw',CFG.seq,CFG.ffn,'')}</div>
        <div class="ag-sec"><h4>Down — OUTPUT [${CFG.seq}×${CFG.d}]</h4>${makeGrid('dn',CFG.seq,CFG.d,'')}</div>
        <div class="ag-sec"><h4>X_out (Residual) — OUTPUT</h4>${makeGrid('rs2',CFG.seq,CFG.d,'')}</div>`,
    // 9: Loss
    ()=>`<div class="ag-sec"><h4>X̂_final (RMSNorm) — OUTPUT</h4>${makeGrid('nf',CFG.seq,CFG.d,'')}</div>
        <div class="ag-sec"><h4>Logits — OUTPUT [${CFG.seq}×${CFG.V}]</h4>${makeGrid('lg',CFG.seq,CFG.V,'')}</div>
        <div class="ag-sec"><h4>Probabilities — OUTPUT</h4>${makeGrid('pb',CFG.seq,CFG.V,'')}</div>
        <div class="ag-sec"><h4>Loss per token</h4><div class="tok-row">${Array.from({length:CFG.seq},(_,i)=>`<span class="mc empty" id="ls-${i}" style="min-width:50px">·</span>`).join('')}</div>
        <div id="tot-loss" style="font-size:.9rem;margin-top:6px;text-align:center">·</div></div>`,
    // 10: Backprop
    ()=>`<div class="ag-sec"><h4>dL/d_logits — OUTPUT [${CFG.seq}×${CFG.V}]</h4>${makeGrid('dl',CFG.seq,CFG.V,'')}</div>`,
    // 11: AdamW
    ()=>`<div class="ag-sec" style="max-width:500px;margin:0 auto"><h4>AdamW Update — W_Q[0][0]</h4>
        <div style="font-size:.75rem;line-height:1.6;padding:8px;background:var(--panel2);border-radius:4px">
        θ (param) = ${fmt(S.adam.param)}<br>g (gradient) = ${S.adam.grad}<br>
        β₁=${CFG.b1}, β₂=${CFG.b2}, lr=${CFG.lr}, λ=${CFG.wd}<br>
        m_prev=${fmt(S.adam.m_prev)}, v_prev=${fmt(S.adam.v_prev)}, step=${S.adam.step}
        </div></div>`
];

// ═══════════════════════════════════════
// PLAYBACK ENGINE
// ═══════════════════════════════════════
let lastSI=-1;

function execStep(i){
    if(i<0||i>=STEPS.length)return;
    cur=i;
    const st=STEPS[i];
    // Stage change?
    if(st.si!==lastSI){lastSI=st.si;el('mat-panel').innerHTML=LAYOUTS[st.si]();}
    // Show math
    el('comp').innerHTML=st.mf();
    // Execute action
    st.af();
    // Auto-scroll comp panel to bottom
    const cp=el('comp');cp.scrollTop=cp.scrollHeight;
    updateUI();
}

function doStep(){if(cur<STEPS.length-1){execStep(cur+1);}else{pause();}}
function play(){playing=true;el('btn-play').textContent='⏸ Pause';runPlay();}
function pause(){playing=false;clearTimeout(ptimer);el('btn-play').textContent='▶ Play';}
function togglePlay(){playing?pause():play();}
function runPlay(){if(!playing||cur>=STEPS.length-1){pause();return;}ptimer=setTimeout(()=>{doStep();runPlay();},spd);}

function skipStage(){
    pause();
    const si=STEPS[cur].si;
    // Execute all remaining steps in current stage
    while(cur<STEPS.length-1&&STEPS[cur+1].si===si){cur++;STEPS[cur].af();}
    // Render last step's layout if stage changed
    if(cur<STEPS.length-1)execStep(cur+1);
    else execStep(cur);
}

function updateUI(){
    const st=STEPS[cur],sm=SMETA[st.si];
    el('stage-name').textContent=sm.name;
    el('step-info').textContent=`Step ${cur+1} / ${STEPS.length}`;
    el('step-label').textContent=st.label;
    const pct=((cur+1)/STEPS.length*100).toFixed(1);
    el('prog-bar').style.width=pct+'%';
    el('prog-pct').textContent=pct+'%';
}

function onSpeed(){spd=210-el('spd-slider').value*20;el('spd-label').textContent=el('spd-slider').value+'x';}

// ═══ INIT ═══
function animInit(){
    initState();
    buildAll();
    el('total-steps').textContent=STEPS.length.toLocaleString();
    execStep(0);
}
