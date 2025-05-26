const form=document.getElementById('profileForm')
if(form){
form.addEventListener('submit',async e=>{
e.preventDefault()
const data=new FormData(form)
const res=await fetch('/start',{method:'POST',body:data})
if(res.ok){location.href='/recommend'}
})
}
async function render(track){
document.getElementById('trackTitle').textContent=track.title
document.getElementById('trackMeta').textContent=`${track.artist} • ${track.genre}`
document.getElementById('spotifyEmbed').innerHTML=track.spotify_iframe
document.getElementById('vkEmbed').innerHTML=track.vk_iframe
}
if(typeof initialData!=='undefined'){render(initialData)}
const nextBtn=document.getElementById('nextBtn')
if(nextBtn){
nextBtn.onclick=async()=>{
const r=await fetch('/next')
if(r.ok){render(await r.json())}
}
}
const genreBtn=document.getElementById('genreBtn')
if(genreBtn){
genreBtn.onclick=async()=>{
const r=await fetch('/by_genre')
if(r.ok){render(await r.json())}
}
}
async function feedback(mark){
await fetch('/feedback',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({mark})})
}
const likeBtn=document.getElementById('likeBtn')
const dislikeBtn=document.getElementById('dislikeBtn')
if(likeBtn){likeBtn.onclick=()=>feedback(1)}
if(dislikeBtn){dislikeBtn.onclick=()=>feedback(0)}
