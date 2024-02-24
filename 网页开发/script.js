document.addEventListener('DOMContentLoaded', function () {
    const chatBody = document.getElementById('chat-body');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');

    // 处理用户点击发送按钮或按下 Enter 键
    sendButton.addEventListener('click', sendMessage);
    userInput.addEventListener('keyup', function (event) {
        if (event.key === 'Enter') {
            sendMessage();
        }
    });

    function sendMessage() {
        const userMessage = userInput.value;
        appendMessage('You', userMessage);

        // 发送用户消息到后端，获取响应（使用Ajax、WebSocket等）
        // 然后将响应添加到聊天窗口中
        userInput.value = ''; // 清空输入框
    }

    function appendMessage(sender, message) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message');
        messageDiv.innerHTML = `<strong>${sender}:</strong> ${message}`;
        chatBody.appendChild(messageDiv);
        chatBody.scrollTop = chatBody.scrollHeight; // 滚动到最新消息
    }
});
