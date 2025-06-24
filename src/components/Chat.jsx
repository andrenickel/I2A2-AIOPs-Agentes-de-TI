import React, { useState, useRef, useEffect } from 'react';

const Chat = () => {
    const [inputMessage, setInputMessage] = useState('');
    const [messages, setMessages] = useState([]);
    const messagesEndRef = useRef(null);

    const webhookUrl = 'https://i2a2-aiops-agentes-de-ti-n8n.bysger.easypanel.host/webhook/f04cbf21-21b2-4777-87ef-6b801b1355a5/chat';
    const sessionId = '28993786-3aca-4b83-8fd9-fbff5ef9ef5c';

    const addMessage = (sender, text, type) => {
        setMessages((prev) => [...prev, { sender, text, type }]);
        setTimeout(() => {
            messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
        }, 100);
    };

    const hasGreeted = useRef(false);

    useEffect(() => {
        if (!hasGreeted.current) {
            addMessage("Assistente", "Olá! Como posso ajudar?", "bot");
            hasGreeted.current = true;
        }
    }, []);

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!inputMessage.trim()) return;

        const userMessage = inputMessage;
        addMessage('Você', userMessage, 'user');

        setInputMessage('');

        try {
            const response = await fetch(webhookUrl, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    action: 'sendMessage',
                    sessionId,
                    chatInput: userMessage
                })
            });

            const data = await response.json();
            addMessage('Assistente', data.output || JSON.stringify(data), 'bot');
        } catch (error) {
            addMessage('Assistente', 'Não foi possível enviar a mensagem.', 'bot');
        }
    };


    return (
        <div className="p-6">
            <div className="mx-auto bg-white rounded-xl shadow-lg p-8">
                <h2 className="text-2xl font-bold text-gray-800 mb-6">Assistente Fiscal IA – Chat</h2>
                <div className="h-96 overflow-y-auto border rounded-md p-4 bg-gray-50 mb-4">
                    {messages.map((msg, i) => (
                        <div key={i} className={`mb-2`}>
                            <span className={`font-semibold ${msg.type === 'user' ? 'text-gray-800' : msg.type === 'bot' ? 'text-green-700' : 'text-red-600'}`}>
                                {msg.sender}:
                            </span>{' '}
                            {msg.text}
                        </div>
                    ))}
                    <div ref={messagesEndRef} />
                </div>

                <form onSubmit={handleSubmit} className="flex gap-4">
                    <input
                        type="text"
                        placeholder="Digite sua mensagem..."
                        className="flex-grow p-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-400"
                        value={inputMessage}
                        onChange={(e) => setInputMessage(e.target.value)}
                        required
                    />
                    <button
                        type="submit"
                        className="bg-green-600 text-white px-4 py-2 rounded-md hover:bg-green-700 transition"
                    >
                        Enviar
                    </button>
                </form>
            </div>
        </div>
    );
}

const styles = {
    body: {
        fontFamily: 'Arial, sans-serif',
        background: '#f4f4f4',
        padding: 20,
        minHeight: '100vh'
    },
    chat: {
        margin: 'auto',
        background: 'white',
        borderRadius: 8,
        padding: 20,
        boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
    },
    messages: {
        overflowY: 'auto',
        border: '1px solid #ccc',
        padding: 10,
        marginBottom: 10
    },
    msg: {
        margin: '5px 0'
    },
    user: {
        color: '#2c3e50'
    },
    bot: {
        color: '#16a085'
    }
};

export default Chat;

