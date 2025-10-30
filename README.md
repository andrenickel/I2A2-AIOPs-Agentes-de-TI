# 🧾 Agente NFe – Sistema Inteligente de Auditoria Fiscal

## 📘 Descrição do Projeto

O **Agente NFe** é um sistema de auditoria fiscal inteligente desenvolvido para **analisar e validar automaticamente notas fiscais eletrônicas (NF-e)**.  
Ele combina **modelos de IA generativa (LLM)**, **pipelines de processamento em Python** e **integrações via API** para automatizar a extração, validação e auditoria de dados fiscais em larga escala.

O sistema foi construído em **duas partes principais**:

1. **Frontend (React + Lovable.dev)**  
   Interface moderna para exibir relatórios, dashboards e resultados de auditoria fiscal.  
   Inclui gráficos de evolução diária, indicadores de desempenho (KPIs) e análise de impostos.

2. **Backend (FastAPI + N8N + IA Generativa)**  
   Contém os agentes de IA responsáveis pelas etapas de:
   - **Extração e ingestão** de arquivos ZIP ou XML de NF-e.  
   - **Validação determinística** e limpeza de dados.  
   - **Auditoria com LLM**, verificando consistência de impostos, CFOP, CST, NCM e divergências fiscais.  
   - **Armazenamento em PostgreSQL**, com tabelas `nfe_notafiscal`, `nf_itens` e `nf_auditoria`.  
   - **Dashboard** de indicadores e comparativos automáticos por período.

O pipeline principal segue o fluxo:

```
extract → validate_clean → load_db → audit_llm → finish
```

---

## ⚙️ Funcionalidades Principais

- 📥 **Ingestão automática** de arquivos ZIP contendo NF-e (cabeçalhos e itens).  
- 🧠 **Auditoria com IA generativa**, sugerindo correções e apontando riscos fiscais.  
- 📊 **Dashboard consolidado**, com KPIs de valor total, documentos processados e taxa de erro.  
- 💾 **Banco de dados PostgreSQL** otimizado com `upsert` em lotes configuráveis (`NFE_BATCH_SIZE`).  
---

## 🧩 Tecnologias Utilizadas

| Camada | Tecnologias |
|--------|--------------|
| **Frontend** | React, Lovable.dev, Tailwind, Chart.js |
| **Backend** | Python, FastAPI, SQLAlchemy, LangGraph |
| **IA e Agentes** | OpenAI GPT (LLM audit), LangChain, N8N automations |
| **Banco de Dados** | PostgreSQL |
| **Infraestrutura** | Docker, Terraform, AWS (opcional) |

---

## 🚀 Como Executar Localmente

### Pré-requisitos
- Python 3.11+
- PostgreSQL
- Node.js 18+
- Docker (opcional)

### Passos

1. Clone o repositório:
   ```bash
   git clone https://github.com/andrenickel/I2A2-AIOPs-Agentes-de-TI.git
   cd I2A2-AIOPs-Agentes-de-TI
   cd backend
   ```

2. Configure o ambiente:
   ```bash
   cp .env.example .env
   # Edite o arquivo com suas credenciais e chaves
   ```

3. Instale as dependências do backend:
   ```bash
   pip install -r requirements.txt
   ```

4. Execute o servidor:
   ```bash
   uvicorn main:app --reload
   ```
---

## 🧑‍💼 Equipe

- **André Amorim**
- **André Nickel**
- **André Pinto**
- **Murilo Ferrari**


---

## 📄 Licença

### MIT License

Copyright (c) 2025 **Equipe AIOPs-Agentes-de-TI**

Por meio desta, é concedida permissão, gratuitamente, a qualquer pessoa que obtenha uma cópia deste software e dos arquivos de documentação associados (o "Software"), para lidar no Software sem restrição, incluindo, sem limitação, os direitos de usar, copiar, modificar, mesclar, publicar, distribuir, sublicenciar e/ou vender cópias do Software, e permitir que pessoas a quem o Software é fornecido o façam, sujeitas às seguintes condições:

A declaração de copyright acima e esta permissão devem ser incluídas em todas as cópias ou partes substanciais do Software.

O SOFTWARE É FORNECIDO "NO ESTADO EM QUE SE ENCONTRA", SEM GARANTIA DE QUALQUER TIPO, EXPRESSA OU IMPLÍCITA, INCLUINDO, MAS NÃO SE LIMITANDO ÀS GARANTIAS DE COMERCIALIZAÇÃO, ADEQUAÇÃO A UM DETERMINADO PROPÓSITO E NÃO VIOLAÇÃO. EM NENHUM CASO OS AUTORES OU DETENTORES DO COPYRIGHT SERÃO RESPONSÁVEIS POR QUALQUER REIVINDICAÇÃO, DANO OU OUTRA RESPONSABILIDADE, SEJA EM AÇÃO DE CONTRATO, DELITO OU DE OUTRA FORMA, DECORRENTE DE, OU EM CONEXÃO COM, O SOFTWARE OU O USO OU OUTRAS NEGOCIAÇÕES NO SOFTWARE.
