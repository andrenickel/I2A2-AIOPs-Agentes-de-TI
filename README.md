# 🧾 Agente NFe – Sistema Inteligente de Auditoria Fiscal

## 📘 Descrição do Projeto

O **Agente NFe** é um sistema de auditoria fiscal inteligente desenvolvido para **analisar e validar automaticamente notas fiscais eletrônicas (NF-e)**.  
Ele combina **modelos de IA generativa (LLM)**, **pipelines de processamento em Python** e **integrações via API** para automatizar a extração, validação e auditoria de dados fiscais em larga escala.

**OBS:** Os Artefatos do produto estão na pasta **Projeto Final - Artefatos**, onde contem Relátorio do Projeto, Apresentações PPTX e MP4 de Pitch do Produto.

O sistema foi construído em **duas partes principais**:

1. **Frontend (React + Lovable.dev)**  
   Interface moderna para exibir relatórios, dashboards e resultados de auditoria fiscal.  
   Inclui gráficos de evolução diária, indicadores de desempenho (KPIs) e análise de impostos.

2. **Backend (Python + N8N + IA Generativa)**  
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

# 📘 Documentação de Instalação e Execução do Projeto

## 1. Clonar o Repositório
```bash
git clone https://github.com/andrenickel/I2A2-AIOPs-Agentes-de-TI.git
cd I2A2-AIOPs-Agentes-de-TI
```

---

## 2. Banco de Dados

1. Crie um banco de dados **PostgreSQL**.  
2. Execute os scripts SQL localizados na pasta **`/sql`** para criar as tabelas e estruturas necessárias.

---

## 3. Backend

### 3.1. Acessar o diretório
```bash
cd backend
```

### 3.2. Configuração de Ambiente
Crie um arquivo `.env` (ou defina variáveis de ambiente) com o seguinte conteúdo:

```bash
DATABASE_URL=postgresql+psycopg2://user:pass@host:5432/db
OPENAI_KEY_API=sjkaj-123124123123
```

### 3.3. Instalar Dependências
```bash
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

### 3.4. Executar o Servidor
```bash
uvicorn main:api --host 0.0.0.0 --port 8000
```

O backend ficará disponível em:  
👉 **https://localhost:8000**

---

## 4. N8N (Automação)

1. Importe os **workflows** da pasta **`/n8n_workflows`** para o N8N.  
2. Configure as **variáveis Secrets** para o banco de dados e para a API da OpenAI.  
3. Atribua essas variáveis aos nós correspondentes dentro dos workflows.  
4. Publique e salve os **endpoints gerados**.

---

## 5. Frontend

### 5.1. Acessar o diretório
```bash
cd frontend
```

### 5.2. Instalar Dependências
```bash
npm install
```

### 5.3. Configurar Endpoints
Atualize os endpoints nos seguintes arquivos:

| Caminho | Linha | API | 
|----------|-------|---------| 
| `/components/Chat.ts` | 37 | API Chat N8N | 
| `/components/FileUpload.ts` | 121 | API /Ingest Python | 
| `/hook/useAIAnalysis.ts` | 60 | API /Analise Python | 
| `/hook/useDashboardData.ts` | 98 | API /Dashboard Python | 
| `/hook/useDocuments.ts` | 72 | API Docs N8N | 
| `/hook/useHomeStats.ts` | 14 | API /Home Python | 

---

### 5.4. Executar o Projeto
```bash
npm run dev
```

O frontend ficará disponível em:  
👉 **https://localhost:8080**

---

## ⚙️ Estrutura Geral do Projeto

```
I2A2-AIOPs-Agentes-de-TI/
├── backend/
│   ├── main.py
│   ├── requirements.txt
│   └── .env
├── frontend/
│   ├── src/
│   ├── package.json
│   └── vite.config.js
├── n8n_workflows/
│   ├── workflow_1.json
│   └── workflow_2.json
└── sql/
    ├── create_tables.sql
    └── seed_data.sql
```

---

## 🧩 Tecnologias Utilizadas
- **Python / FastAPI** — Backend e APIs  
- **PostgreSQL** — Banco de Dados  
- **N8N** — Automação e integração entre agentes  
- **React + Vite + TypeScript** — Frontend  
- **OpenAI API** — IA generativa para agentes inteligentes  

---

## 🚀 Execução Completa

1. Subir o banco de dados.  
2. Iniciar o backend (`uvicorn`).  
3. Importar e ativar os workflows no N8N.  
4. Rodar o frontend (`npm run dev`).  

Após isso, o sistema estará operacional e integrado entre as três camadas.

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
