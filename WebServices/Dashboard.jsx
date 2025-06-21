import React, { useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, LineChart, Line } from 'recharts';
import { TrendingUp, TrendingDown, DollarSign, FileText, AlertTriangle, CheckCircle } from 'lucide-react';

const Dashboard = () => {
  const [selectedPeriod, setSelectedPeriod] = useState('month');

  // Mock data for charts
  const monthlyData = [
    { name: 'Jan', documentos: 65, valor: 125000 },
    { name: 'Fev', documentos: 78, valor: 145000 },
    { name: 'Mar', documentos: 90, valor: 180000 },
    { name: 'Abr', documentos: 81, valor: 165000 },
    { name: 'Mai', documentos: 95, valor: 195000 },
    { name: 'Jun', documentos: 102, valor: 210000 },
  ];

  const supplierData = [
    { name: 'Fornecedor A', value: 35, color: '#2E8B57' },
    { name: 'Fornecedor B', value: 25, color: '#1E90FF' },
    { name: 'Fornecedor C', value: 20, color: '#FFD700' },
    { name: 'Fornecedor D', value: 15, color: '#FF6B6B' },
    { name: 'Outros', value: 5, color: '#95A5A6' },
  ];

  const taxData = [
    { name: 'ICMS', valor: 45000, percentual: 18 },
    { name: 'IPI', valor: 12000, percentual: 5 },
    { name: 'PIS', valor: 8000, percentual: 1.65 },
    { name: 'COFINS', valor: 15000, percentual: 7.6 },
  ];

  return (
    <div className="space-y-6">
      {/* Header with period selector */}
      <div className="flex justify-between items-center">
        <h2 className="text-3xl font-bold text-white">Dashboard Fiscal</h2>
        <select 
          value={selectedPeriod}
          onChange={(e) => setSelectedPeriod(e.target.value)}
          className="bg-white/20 text-white border border-white/30 rounded-lg px-4 py-2 backdrop-blur-md"
        >
          <option value="week">Esta Semana</option>
          <option value="month">Este Mês</option>
          <option value="quarter">Este Trimestre</option>
          <option value="year">Este Ano</option>
        </select>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="fiscal-card rounded-xl p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Total de Documentos</p>
              <p className="text-2xl font-bold fiscal-green">1,247</p>
              <div className="flex items-center text-green-600 text-sm">
                <TrendingUp className="w-4 h-4 mr-1" />
                +12% vs mês anterior
              </div>
            </div>
            <FileText className="w-12 h-12 fiscal-green" />
          </div>
        </div>

        <div className="fiscal-card rounded-xl p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Valor Total</p>
              <p className="text-2xl font-bold fiscal-blue">R$ 2.1M</p>
              <div className="flex items-center text-blue-600 text-sm">
                <TrendingUp className="w-4 h-4 mr-1" />
                +8% vs mês anterior
              </div>
            </div>
            <DollarSign className="w-12 h-12 fiscal-blue" />
          </div>
        </div>

        <div className="fiscal-card rounded-xl p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Taxa de Erro</p>
              <p className="text-2xl font-bold text-red-500">2.3%</p>
              <div className="flex items-center text-red-500 text-sm">
                <TrendingDown className="w-4 h-4 mr-1" />
                -0.5% vs mês anterior
              </div>
            </div>
            <AlertTriangle className="w-12 h-12 text-red-500" />
          </div>
        </div>

        <div className="fiscal-card rounded-xl p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Processamento</p>
              <p className="text-2xl font-bold fiscal-green">97.7%</p>
              <div className="flex items-center text-green-600 text-sm">
                <CheckCircle className="w-4 h-4 mr-1" />
                Automático
              </div>
            </div>
            <CheckCircle className="w-12 h-12 fiscal-green" />
          </div>
        </div>
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Monthly Documents Chart */}
        <div className="fiscal-card rounded-xl p-6">
          <h3 className="text-xl font-semibold mb-4">Documentos por Mês</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={monthlyData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="documentos" fill="#2E8B57" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Supplier Distribution */}
        <div className="fiscal-card rounded-xl p-6">
          <h3 className="text-xl font-semibold mb-4">Distribuição por Fornecedor</h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={supplierData}
                cx="50%"
                cy="50%"
                outerRadius={100}
                fill="#8884d8"
                dataKey="value"
                label={({ name, value }) => `${name}: ${value}%`}
              >
                {supplierData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Revenue Chart */}
      <div className="fiscal-card rounded-xl p-6">
        <h3 className="text-xl font-semibold mb-4">Evolução do Valor Total</h3>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={monthlyData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" />
            <YAxis />
            <Tooltip formatter={(value) => [`R$ ${value.toLocaleString()}`, 'Valor']} />
            <Line type="monotone" dataKey="valor" stroke="#1E90FF" strokeWidth={3} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Tax Summary */}
      <div className="fiscal-card rounded-xl p-6">
        <h3 className="text-xl font-semibold mb-4">Resumo de Impostos</h3>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          {taxData.map((tax, index) => (
            <div key={index} className="bg-gray-50 rounded-lg p-4">
              <h4 className="font-semibold text-gray-800">{tax.name}</h4>
              <p className="text-2xl font-bold fiscal-green">R$ {tax.valor.toLocaleString()}</p>
              <p className="text-sm text-gray-600">{tax.percentual}% alíquota média</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default Dashboard;

