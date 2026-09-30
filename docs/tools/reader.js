/* Offline reader: local index, no fetch or remote dependencies. */
(()=>{'use strict';
const $=s=>document.querySelector(s), $$=s=>[...document.querySelectorAll(s)];
const get=(k,f)=>{try{return localStorage.getItem(k)||f}catch{return f}};
const put=(k,v)=>{try{localStorage.setItem(k,v)}catch{}};
const root=document.documentElement;
root.dataset.theme=get('deepep-theme','light');
let font=Number(get('deepep-font','17')); root.style.setProperty('--reader-size',font+'px');
$('#theme')?.addEventListener('click',()=>{root.dataset.theme=root.dataset.theme==='dark'?'light':'dark';put('deepep-theme',root.dataset.theme)});
$$('.font-btn').forEach(b=>b.addEventListener('click',()=>{font=Math.max(14,Math.min(22,font+Number(b.dataset.step)));root.style.setProperty('--reader-size',font+'px');put('deepep-font',String(font))}));
$('#print')?.addEventListener('click',()=>window.print());
$('#menu')?.addEventListener('click',()=>{const open=$('.sidebar').classList.toggle('open');$('#menu').setAttribute('aria-expanded',String(open))});
$('#top')?.addEventListener('click',()=>window.scrollTo({top:0,behavior:'smooth'}));
const toast=t=>{let d=document.createElement('div');d.className='status-toast';d.role='status';d.textContent=t;document.body.append(d);setTimeout(()=>d.remove(),1600)};
$$('.copy').forEach(b=>b.addEventListener('click',async()=>{const text=b.closest('.code-block').querySelector('code').textContent;try{if(navigator.clipboard){await navigator.clipboard.writeText(text)}else{throw Error('fallback')}toast('代码已复制')}catch{const a=document.createElement('textarea');a.value=text;document.body.append(a);a.select();const ok=document.execCommand('copy');a.remove();toast(ok?'代码已复制':'请选中代码后复制')}}));
const progress=()=>{const total=document.documentElement.scrollHeight-innerHeight;$('.progress').style.width=(total>0?scrollY/total*100:0)+'%'};
addEventListener('scroll',progress,{passive:true});progress();
const heads=$$('.article h2[id],.article h3[id]');
if('IntersectionObserver' in window){const observer=new IntersectionObserver(entries=>{for(const e of entries){if(e.isIntersecting){$$('.toc a').forEach(a=>a.classList.toggle('current',a.getAttribute('href')==='#'+e.target.id))}}},{rootMargin:'-80px 0px -65% 0px'});heads.forEach(h=>observer.observe(h))}
const dialog=$('#search'), input=$('#query'), results=$('#results'), info=$('#search-info');
const openSearch=()=>{if(!dialog.open)dialog.showModal();input.focus()};
$('#open-search')?.addEventListener('click',openSearch);
$('#close-search')?.addEventListener('click',()=>dialog.close());
dialog?.addEventListener('click',e=>{if(e.target===dialog)dialog.close()});
addEventListener('keydown',e=>{if((e.ctrlKey||e.metaKey)&&e.key.toLowerCase()==='k'){e.preventDefault();openSearch()}});
const esc=t=>t.replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function highlight(text,terms){let safe=esc(text);for(const term of terms){const q=esc(term).replace(/[.*+?^${}()|[\]\\]/g,'\\$&');safe=safe.replace(new RegExp(q,'gi'),m=>'<mark>'+m+'</mark>')}return safe}
function search(){const q=input.value.trim().toLowerCase();if(!q){results.innerHTML='';info.textContent='输入中文概念、函数名或参数；多个词用空格分隔。';return}const terms=q.split(/\s+/).filter(Boolean);let matches=(window.DEEPEP_SEARCH||[]).map(x=>{const full=(x.title+' '+x.doc+' '+x.text).toLowerCase();if(!terms.every(t=>full.includes(t)))return null;const score=terms.reduce((s,t)=>s+(x.title.toLowerCase().includes(t)?10:0)+(x.doc.toLowerCase().includes(t)?3:0),0);return {...x,score}}).filter(Boolean).sort((a,b)=>b.score-a.score);info.textContent=`找到 ${matches.length} 个章节，显示前 ${Math.min(matches.length,45)} 个。`;results.innerHTML=matches.slice(0,45).map(x=>{const ix=x.text.toLowerCase().indexOf(terms[0]);const start=Math.max(0,ix-45);const excerpt=x.text.slice(start,start+170);return `<a class="result" href="${esc(x.url)}"><small>${esc(x.doc)}</small><strong>${highlight(x.title,terms)}</strong><p>${start?'…':''}${highlight(excerpt,terms)}…</p></a>`}).join('');}
input?.addEventListener('input',search);results?.addEventListener('click',e=>{if(e.target.closest('.result'))dialog.close()});search();
// Local file fragments use UTF-8 titles and source line IDs.
if(location.hash){setTimeout(()=>{let id;try{id=decodeURIComponent(location.hash.slice(1))}catch{return}document.getElementById(id)?.scrollIntoView()},80)}
})();
