import { useState, useEffect, useRef, useCallback, memo } from 'react';
import { Send, Download, Bot, User, Copy, Trash2, ArrowDown, AlertCircle, CheckCircle } from 'lucide-react';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Card } from './ui/card';
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from './ui/dropdown-menu';
import { useToast } from '@/hooks/use-toast';
import { Textarea } from './ui/textarea';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from './ui/tooltip';
import ReactMarkdown from 'react-markdown';
import jsPDF from 'jspdf';

interface Message {
  sender: 'user' | 'bot';
  text: string;
  timestamp: Date;
  isError?: boolean;
}

const SUGGESTED_QUESTIONS = [
  "Qual o valor total das notas ficais",
  "Qual nota com o maior valor",
  "Qual fornecedor com maior número de notas",
  "Quantas notas emitidas no trimestre"
];

const Chat = memo(() => {
  const [inputMessage, setInputMessage] = useState('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [isTyping, setIsTyping] = useState(false);
  const [isSending, setIsSending] = useState(false);
  const [showScrollButton, setShowScrollButton] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const messagesContainerRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const { toast } = useToast();
  const webhookUrl = 'https://i2a2-aiops-agentes-de-ti-n8n.bysger.easypanel.host/webhook/f04cbf21-21b2-4777-87ef-6b801b1355a5/chat';
  const sessionId = '28993786-3aca-4b83-8fd9-fbff5ef9ef5c';
  const MAX_CHARS = 2000;

  const scrollToBottom = useCallback((smooth = true) => {
    messagesEndRef.current?.scrollIntoView({ behavior: smooth ? 'smooth' : 'auto' });
  }, []);

  const addMessage = useCallback((message: Message) => {
    setMessages((prev) => [...prev, message]);
    setTimeout(scrollToBottom, 100);
  }, [scrollToBottom]);

  const handleScroll = useCallback(() => {
    if (!messagesContainerRef.current) return;
    const { scrollTop, scrollHeight, clientHeight } = messagesContainerRef.current;
    setShowScrollButton(scrollHeight - scrollTop - clientHeight > 100);
  }, []);

  const copyToClipboard = useCallback((text: string) => {
    navigator.clipboard.writeText(text);
    toast({
      title: "Copiado",
      description: "Mensagem copiada para a área de transferência."
    });
  }, [toast]);

  const clearChat = useCallback(() => {
    setMessages([{ 
      sender: 'bot', 
      text: 'Olá! Como posso te ajudar hoje, com suas notas fiscais?',
      timestamp: new Date()
    }]);
    toast({
      title: "Chat limpo",
      description: "Histórico de conversa foi apagado."
    });
  }, [toast]);

  const handleSuggestedQuestion = useCallback((question: string) => {
    setInputMessage(question);
    textareaRef.current?.focus();
  }, []);

  useEffect(() => {
    addMessage({ 
      sender: 'bot', 
      text: 'Olá! Como posso te ajudar hoje, com suas notas fiscais?',
      timestamp: new Date()
    });
  }, [addMessage]);

  useEffect(() => {
    const container = messagesContainerRef.current;
    if (container) {
      container.addEventListener('scroll', handleScroll);
      return () => container.removeEventListener('scroll', handleScroll);
    }
  }, [handleScroll]);

  const exportToCSV = useCallback(() => {
    if (messages.length <= 1) {
      toast({
        title: "Nenhuma mensagem",
        description: "Não há mensagens para exportar.",
        variant: "destructive"
      });
      return;
    }

    const csvContent = [
      ['Remetente', 'Mensagem', 'Horário'],
      ...messages.map((msg) => [
        msg.sender === 'user' ? 'Usuário' : 'IA Fiscal',
        msg.text.replace(/"/g, '""'),
        msg.timestamp.toLocaleString('pt-BR')
      ])
    ]
      .map(row => row.map(cell => `"${cell}"`).join(','))
      .join('\n');

    const blob = new Blob(['\ufeff' + csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = `conversa_${new Date().toISOString().split('T')[0]}.csv`;
    link.click();

    toast({
      title: "Exportado com sucesso",
      description: "Conversa exportada em formato CSV."
    });
  }, [messages, toast]);

  const exportToPDF = useCallback(() => {
    if (messages.length <= 1) {
      toast({
        title: "Nenhuma mensagem",
        description: "Não há mensagens para exportar.",
        variant: "destructive"
      });
      return;
    }

    const doc = new jsPDF();
    const pageWidth = doc.internal.pageSize.getWidth();
    const margin = 15;
    const maxWidth = pageWidth - (margin * 2);
    let yPosition = 20;

    doc.setFontSize(16);
    doc.text('Conversa - AIOps Fiscal Agent', margin, yPosition);
    yPosition += 10;

    doc.setFontSize(10);
    doc.text(`Data: ${new Date().toLocaleString('pt-BR')}`, margin, yPosition);
    yPosition += 15;

    doc.setFontSize(11);
    messages.forEach((msg) => {
      const sender = msg.sender === 'user' ? 'Usuário' : 'IA Fiscal';
      const lines = doc.splitTextToSize(`${sender}: ${msg.text}`, maxWidth);
      
      if (yPosition + (lines.length * 7) > doc.internal.pageSize.getHeight() - 20) {
        doc.addPage();
        yPosition = 20;
      }

      doc.setFont(undefined, msg.sender === 'user' ? 'bold' : 'normal');
      doc.text(lines, margin, yPosition);
      yPosition += (lines.length * 7) + 5;
    });

    doc.save(`conversa_${new Date().toISOString().split('T')[0]}.pdf`);

    toast({
      title: "Exportado com sucesso",
      description: "Conversa exportada em formato PDF."
    });
  }, [messages, toast]);

  const handleSubmit = useCallback(async (e?: React.FormEvent) => {
    e?.preventDefault();
    if (!inputMessage.trim() || isSending || inputMessage.length > MAX_CHARS) return;

    const userMessage = inputMessage;
    setInputMessage('');
    setIsSending(true);
    addMessage({ sender: 'user', text: userMessage, timestamp: new Date() });

    // Reset textarea height
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }

    setIsTyping(true);

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
      setIsTyping(false);
      addMessage({ 
        sender: 'bot', 
        text: data.output || 'Desculpe, não consegui processar sua mensagem.',
        timestamp: new Date()
      });
    } catch (error) {
      console.error('Error sending message:', error);
      setIsTyping(false);
      addMessage({ 
        sender: 'bot', 
        text: 'Desculpe, ocorreu um erro ao processar sua mensagem.',
        timestamp: new Date(),
        isError: true
      });
    } finally {
      setIsSending(false);
    }
  }, [inputMessage, isSending, addMessage, webhookUrl, sessionId]);

  const handleKeyDown = useCallback((e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  }, [handleSubmit]);

  const handleInputChange = useCallback((e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInputMessage(e.target.value);
    // Auto-resize textarea
    e.target.style.height = 'auto';
    e.target.style.height = Math.min(e.target.scrollHeight, 120) + 'px';
  }, []);

  return (
    <TooltipProvider>
      <div className="h-[calc(100vh-12rem)] flex flex-col space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="text-3xl font-bold text-white">Chat com IA Fiscal</h2>
          <Button
            variant="outline"
            size="sm"
            onClick={clearChat}
            className="gap-2"
            aria-label="Limpar chat"
          >
            <Trash2 className="w-4 h-4" />
            Limpar
          </Button>
        </div>

        <Card className="bg-card/95 backdrop-blur-sm flex-1 flex flex-col border-border/50 relative">
          {/* Messages Area */}
          <div 
            ref={messagesContainerRef}
            className="flex-1 overflow-y-auto p-6 space-y-4"
            role="log"
            aria-live="polite"
            aria-label="Mensagens do chat"
          >
            {messages.length === 1 && (
              <div className="space-y-4 mb-6">
                <p className="text-muted-foreground text-center mb-4">Perguntas sugeridas:</p>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                  {SUGGESTED_QUESTIONS.map((question, idx) => (
                    <Button
                      key={idx}
                      variant="outline"
                      className="text-left h-auto py-3 px-4 whitespace-normal hover-scale"
                      onClick={() => handleSuggestedQuestion(question)}
                    >
                      {question}
                    </Button>
                  ))}
                </div>
              </div>
            )}

            {messages.map((message, index) => (
              <div
                key={index}
                className={`flex gap-3 animate-fade-in ${message.sender === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                {message.sender === 'bot' && (
                  <div className="flex-shrink-0 w-8 h-8 rounded-full bg-primary/20 flex items-center justify-center">
                    <Bot className="w-5 h-5 text-primary" aria-hidden="true" />
                  </div>
                )}
                
                <div className={`flex flex-col ${message.sender === 'user' ? 'items-end' : 'items-start'} max-w-[70%]`}>
                  <div
                    className={`rounded-lg p-4 ${
                      message.sender === 'user'
                        ? 'bg-gradient-to-r from-primary to-secondary text-primary-foreground'
                        : message.isError
                        ? 'bg-destructive/10 border border-destructive/50 text-foreground'
                        : 'bg-muted text-foreground'
                    }`}
                  >
                    {message.isError && (
                      <div className="flex items-center gap-2 mb-2 text-destructive">
                        <AlertCircle className="w-4 h-4" />
                        <span className="text-sm font-medium">Erro</span>
                      </div>
                    )}
                    {message.sender === 'bot' ? (
                      <ReactMarkdown 
                        components={{
                          p: ({children}) => <p className="mb-2 last:mb-0">{children}</p>,
                          ul: ({children}) => <ul className="list-disc ml-4 mb-2">{children}</ul>,
                          ol: ({children}) => <ol className="list-decimal ml-4 mb-2">{children}</ol>,
                          li: ({children}) => <li className="mb-1">{children}</li>,
                          code: ({children}) => <code className="bg-muted px-1 rounded text-sm">{children}</code>,
                          strong: ({children}) => <strong className="font-semibold">{children}</strong>,
                        }}
                      >
                        {message.text}
                      </ReactMarkdown>
                    ) : (
                      <p className="whitespace-pre-wrap">{message.text}</p>
                    )}
                  </div>
                  
                  <div className="flex items-center gap-2 mt-1">
                    <span className="text-xs text-muted-foreground">
                      {message.timestamp.toLocaleTimeString('pt-BR', { hour: '2-digit', minute: '2-digit' })}
                    </span>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <Button
                          variant="ghost"
                          size="sm"
                          className="h-6 w-6 p-0"
                          onClick={() => copyToClipboard(message.text)}
                          aria-label="Copiar mensagem"
                        >
                          <Copy className="w-3 h-3" />
                        </Button>
                      </TooltipTrigger>
                      <TooltipContent>Copiar mensagem</TooltipContent>
                    </Tooltip>
                  </div>
                </div>

                {message.sender === 'user' && (
                  <div className="flex-shrink-0 w-8 h-8 rounded-full bg-secondary/20 flex items-center justify-center">
                    <User className="w-5 h-5 text-secondary" aria-hidden="true" />
                  </div>
                )}
              </div>
            ))}

            {isTyping && (
              <div className="flex gap-3 animate-fade-in">
                <div className="flex-shrink-0 w-8 h-8 rounded-full bg-primary/20 flex items-center justify-center">
                  <Bot className="w-5 h-5 text-primary" />
                </div>
                <div className="bg-muted rounded-lg p-4">
                  <div className="flex gap-1">
                    <div className="w-2 h-2 bg-foreground/40 rounded-full animate-pulse" style={{ animationDelay: '0ms' }} />
                    <div className="w-2 h-2 bg-foreground/40 rounded-full animate-pulse" style={{ animationDelay: '150ms' }} />
                    <div className="w-2 h-2 bg-foreground/40 rounded-full animate-pulse" style={{ animationDelay: '300ms' }} />
                  </div>
                </div>
              </div>
            )}
            
            <div ref={messagesEndRef} />
          </div>

          {/* Scroll to bottom button */}
          {showScrollButton && (
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="secondary"
                  size="icon"
                  className="absolute bottom-24 right-6 rounded-full shadow-lg animate-scale-in z-10"
                  onClick={() => scrollToBottom()}
                  aria-label="Rolar para o final"
                >
                  <ArrowDown className="w-4 h-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>Rolar para o final</TooltipContent>
            </Tooltip>
          )}

          {/* Input Area */}
          <div className="border-t border-border p-4">
            <form onSubmit={handleSubmit} className="flex gap-2 items-end">
              <div className="flex-1 relative">
                <Textarea
                  ref={textareaRef}
                  value={inputMessage}
                  onChange={handleInputChange}
                  onKeyDown={handleKeyDown}
                  placeholder="Digite sua mensagem... (Shift+Enter para quebrar linha)"
                  className="min-h-[44px] max-h-[120px] resize-none pr-16"
                  aria-label="Campo de mensagem"
                  maxLength={MAX_CHARS}
                  disabled={isSending}
                />
                <div className={`absolute bottom-2 right-2 text-xs ${
                  inputMessage.length > MAX_CHARS * 0.9 ? 'text-destructive' : 'text-muted-foreground'
                }`}>
                  {inputMessage.length}/{MAX_CHARS}
                </div>
              </div>
              <Button
                type="submit"
                className="bg-gradient-to-r from-primary to-secondary"
                disabled={!inputMessage.trim() || isSending || inputMessage.length > MAX_CHARS}
                aria-label="Enviar mensagem"
              >
                {isSending ? (
                  <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                ) : (
                  <>
                    <Send className="w-4 h-4 mr-2" />
                    Enviar
                  </>
                )}
              </Button>
            </form>
          </div>
        </Card>

        <div className="flex justify-end gap-2">
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="outline" size="sm" className="gap-2">
                <Download className="w-4 h-4" />
                Exportar
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="bg-popover z-50">
              <DropdownMenuItem onClick={exportToCSV}>
                Exportar como CSV
              </DropdownMenuItem>
              <DropdownMenuItem onClick={exportToPDF}>
                Exportar como PDF
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </div>
    </TooltipProvider>
  );
});

Chat.displayName = 'Chat';

export default Chat;
