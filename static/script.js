async function sendMessage(){

let message=document.getElementById("message").value;

if(message=="") return;

let chatbox=document.getElementById("chatbox");

chatbox.innerHTML+="<div class='user'><b>You:</b> "+message+"</div>";

document.getElementById("message").value="";

let response=await fetch("/chat",{

method:"POST",

headers:{

"Content-Type":"application/json"

},

body:JSON.stringify({

message:message

})

});

let data=await response.json();

chatbox.innerHTML+="<div class='bot'><b>Bot:</b> "+data.reply+"</div>";

chatbox.scrollTop=chatbox.scrollHeight;

}