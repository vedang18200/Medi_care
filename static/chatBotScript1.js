const chat_btn = $("#chat-bot .icon");
const chat_box = $("#chat-bot .messenger");

chat_btn.click(() => {
  chat_btn.toggleClass("expanded");
  setTimeout(() => {
    chat_box.toggleClass("expanded");
  }, 100);
});