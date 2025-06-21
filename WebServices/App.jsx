import React, { useState } from 'react';
import { FileText, BarChart3, Upload, Home, Settings, LogOut, Menu, X } from 'lucide-react';
import Dashboard from './components/Dashboard';
import FileUpload from './components/FileUpload';
import { NotificationSystem, SearchAndFilter } from './components/NotificationSystem';
import './App.css';

function App() {
  const [activeTab, setActiveTab] = useState('home');
  const [sidebarOpen, setSidebarOpen] = useState(false);

  const navigation = [
    { id: 'home', name: 'Início', icon: Home },
    { id: 'upload', name: 'Upload', icon: Upload },
    { id: 'dashboard', name: 'Dashboard', icon: BarChart3 },
    { id: 'documents', name: 'Documentos', icon: FileText },
  ];

  const handleSearch = (searchTerm) => {
    console.log('Searching for:', searchTerm);
    // Implementar lógica de busca
  };

  const handleFilter = (filterStatus) => {
    console.log('Filtering by:', filterStatus);
    // Implementar lógica de filtro
  };

  const renderContent = () => {
    switch (activeTab) {
      case 'upload':
        return <FileUpload />;
      case 'dashboard':
        return <Dashboard />;
      case 'documents':
        return (
          <div className="space-y-6">
            <h2 className="text-3xl font-bold text-white">Documentos Fiscais</h2>
            
            {/* Search and Filter */}
            <div className="fiscal-card rounded-xl p-6">
              <SearchAndFilter onSearch={handleSearch} onFilter={handleFilter} />
            </div>

            <div className="fiscal-card rounded-xl p-8 text-center">
              <FileText className="w-16 h-16 fiscal-green mx-auto mb-4" />
              <h3 className="text-xl font-semibold mb-2">Biblioteca de Documentos</h3>
              <p className="text-gray-600">Visualize e gerencie todos os seus documentos fiscais processados</p>
              
              {/* Sample document list */}
              <div className="mt-8 space-y-4">
                <div className="bg-green-50 border border-green-200 rounded-lg p-4 text-left">
                  <div className="flex items-center justify-between">
                    <div>
                      <h4 className="font-semibold text-gray-800">NFe_12345678901234567890.xml</h4>
                      <p className="text-sm text-gray-600">Empresa ABC Ltda - R$ 15.750,00</p>
                    </div>
                    <span className="px-3 py-1 bg-green-100 text-green-700 rounded-full text-sm">Processado</span>
                  </div>
                </div>
                
                <div className="bg-orange-50 border border-orange-200 rounded-lg p-4 text-left">
                  <div className="flex items-center justify-between">
                    <div>
                      <h4 className="font-semibold text-gray-800">NFe_09876543210987654321.xml</h4>
                      <p className="text-sm text-gray-600">Fornecedor XYZ S.A. - R$ 8.320,50</p>
                    </div>
                    <span className="px-3 py-1 bg-orange-100 text-orange-700 rounded-full text-sm">Pendente</span>
                  </div>
                </div>
                
                <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-left">
                  <div className="flex items-center justify-between">
                    <div>
                      <h4 className="font-semibold text-gray-800">NFe_11111111111111111111.xml</h4>
                      <p className="text-sm text-gray-600">Empresa DEF ME - R$ 2.100,00</p>
                    </div>
                    <span className="px-3 py-1 bg-red-100 text-red-700 rounded-full text-sm">Erro</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        );
      default:
        return (
          <div className="space-y-6">
            {/* Hero Section */}
            <section className="text-center mb-12">
              <h2 className="text-4xl font-bold text-white mb-4">
                Automatize Sua Gestão Fiscal com IA
              </h2>
              <p className="text-xl text-white/90 mb-8 max-w-2xl mx-auto">
                Processe documentos XML de NF-e em segundos, com validação inteligente e relatórios automáticos
              </p>
              <div className="flex justify-center space-x-4">
                <button 
                  onClick={() => setActiveTab('upload')}
                  className="bg-white text-green-600 px-8 py-3 rounded-lg font-semibold hover:bg-green-50 transition-colors"
                >
                  Começar Agora
                </button>
                <button 
                  onClick={() => setActiveTab('dashboard')}
                  className="border-2 border-white text-white px-8 py-3 rounded-lg font-semibold hover:bg-white/10 transition-colors"
                >
                  Ver Dashboard
                </button>
              </div>
            </section>

            {/* Features Grid */}
            <section className="grid md:grid-cols-3 gap-8 mb-12">
              <div className="fiscal-card rounded-xl p-6 text-center">
                <Upload className="w-12 h-12 fiscal-green mx-auto mb-4" />
                <h3 className="text-xl font-semibold mb-3">Upload Simples</h3>
                <p className="text-gray-600">Faça upload de arquivos XML de NF-e e deixe nossa IA processar automaticamente</p>
              </div>
              
              <div className="fiscal-card rounded-xl p-6 text-center">
                <FileText className="w-12 h-12 fiscal-blue mx-auto mb-4" />
                <h3 className="text-xl font-semibold mb-3">Validação Inteligente</h3>
                <p className="text-gray-600">Validação automática de dados com IA, detectando inconsistências e erros</p>
              </div>
              
              <div className="fiscal-card rounded-xl p-6 text-center">
                <BarChart3 className="w-12 h-12 fiscal-green mx-auto mb-4" />
                <h3 className="text-xl font-semibold mb-3">Relatórios Automáticos</h3>
                <p className="text-gray-600">Dashboards e relatórios gerados automaticamente com insights acionáveis</p>
              </div>
            </section>

            {/* Status Cards */}
            <section className="grid md:grid-cols-3 gap-6 mb-12">
              <div className="fiscal-card rounded-xl p-6">
                <div className="flex items-center justify-between mb-4">
                  <h4 className="font-semibold">Documentos Processados</h4>
                  <FileText className="w-6 h-6 text-green-600" />
                </div>
                <div className="text-3xl font-bold fiscal-green">1,247</div>
                <p className="text-sm text-gray-600">Este mês</p>
              </div>
              
              <div className="fiscal-card rounded-xl p-6">
                <div className="flex items-center justify-between mb-4">
                  <h4 className="font-semibold">Validações Pendentes</h4>
                  <BarChart3 className="w-6 h-6 text-orange-500" />
                </div>
                <div className="text-3xl font-bold text-orange-500">12</div>
                <p className="text-sm text-gray-600">Requer atenção</p>
              </div>
              
              <div className="fiscal-card rounded-xl p-6">
                <div className="flex items-center justify-between mb-4">
                  <h4 className="font-semibold">Taxa de Sucesso</h4>
                  <FileText className="w-6 h-6 text-green-600" />
                </div>
                <div className="text-3xl font-bold fiscal-green">97.7%</div>
                <p className="text-sm text-gray-600">Processamento automático</p>
              </div>
            </section>
          </div>
        );
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-600 to-blue-600">
      {/* Header */}
      <header className="bg-white/10 backdrop-blur-md border-b border-white/20">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <img src="/src/assets/logo.png" alt="Logo" className="w-12 h-12" />
              <div>
                <h1 className="text-xl font-bold text-white">Agente de IA Fiscal</h1>
                <p className="text-sm text-white/80">Automatização de Documentos Fiscais</p>
              </div>
            </div>
            
            <div className="flex items-center space-x-4">
              {/* Notification System */}
              <NotificationSystem />
              
              {/* Desktop Navigation */}
              <nav className="hidden md:flex space-x-6">
                {navigation.map((item) => (
                  <button
                    key={item.id}
                    onClick={() => setActiveTab(item.id)}
                    className={`flex items-center space-x-2 px-3 py-2 rounded-lg transition-colors ${
                      activeTab === item.id
                        ? 'bg-white/20 text-white'
                        : 'text-white/80 hover:text-white hover:bg-white/10'
                    }`}
                  >
                    <item.icon className="w-4 h-4" />
                    <span>{item.name}</span>
                  </button>
                ))}
              </nav>

              {/* Mobile menu button */}
              <button
                onClick={() => setSidebarOpen(!sidebarOpen)}
                className="md:hidden text-white p-2"
              >
                {sidebarOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Mobile Sidebar */}
      {sidebarOpen && (
        <div className="md:hidden fixed inset-0 z-50 bg-black/50" onClick={() => setSidebarOpen(false)}>
          <div className="fixed left-0 top-0 h-full w-64 bg-white/95 backdrop-blur-md p-6">
            <div className="flex items-center justify-between mb-8">
              <h2 className="text-lg font-semibold">Menu</h2>
              <button onClick={() => setSidebarOpen(false)}>
                <X className="w-6 h-6" />
              </button>
            </div>
            <nav className="space-y-4">
              {navigation.map((item) => (
                <button
                  key={item.id}
                  onClick={() => {
                    setActiveTab(item.id);
                    setSidebarOpen(false);
                  }}
                  className={`flex items-center space-x-3 w-full px-3 py-2 rounded-lg transition-colors ${
                    activeTab === item.id
                      ? 'bg-green-100 text-green-700'
                      : 'text-gray-700 hover:bg-gray-100'
                  }`}
                >
                  <item.icon className="w-5 h-5" />
                  <span>{item.name}</span>
                </button>
              ))}
            </nav>
          </div>
        </div>
      )}

      {/* Main Content */}
      <main className="container mx-auto px-4 py-8">
        {renderContent()}
      </main>

      {/* Footer */}
      <footer className="bg-white/10 backdrop-blur-md border-t border-white/20 mt-16">
        <div className="container mx-auto px-4 py-8">
          <div className="text-center text-white/80">
            <p>&copy; 2024 Agente de IA para Automatização Fiscal. Protótipo desenvolvido para demonstração.</p>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;

