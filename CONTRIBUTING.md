# Guia de Contribuição — VIPER

Agradecemos o seu interesse em contribuir com o **VIPER** (*Vehicle Identification and Plate Extraction Recognition*)!

Este documento contém orientações e boas práticas para colaborar com o desenvolvimento do projeto.

---

## 🛠️ Configuração do Ambiente de Desenvolvimento

1. **Faça um Fork do repositório** para a sua conta GitHub.
2. **Clone** o seu fork localmente:
   ```bash
   git clone https://github.com/SEU_USUARIO/VIPER.git
   cd VIPER
   ```
3. **Crie e ative um ambiente virtual**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # No Windows: .venv\Scripts\activate
   ```
4. **Instale as dependências de desenvolvimento**:
   ```bash
   make install-dev
   # ou: pip install -r requirements.txt -r requirements-dev.txt
   ```

---

## 🧪 Executando Testes

Antes de enviar qualquer alteração, certifique-se de que todos os testes unitários continuam passando:

```bash
make test
# ou: pytest -v
```

Você também pode validar a sintaxe e compilação de todos os scripts:

```bash
python -m py_compile main.py preProcessing.py reading.py trainYOLO.py validadeYOLO.py gerar_pdf_relatorio.py src/*.py
```

---

## 🌿 Padrão de Branches

Recomendamos utilizar nomes descritivos para suas branches:

- `feature/nome-da-funcionalidade`: para novas funcionalidades
- `fix/descricao-do-bug`: para correções de problemas
- `docs/melhoria-documentacao`: para atualizações no README ou documentações
- `refactor/modulo-alvo`: para refatoração e otimização de código

---

## 📋 Boas Práticas de Código

- Mantenha o código no padrão PEP 8.
- Escreva docstrings explicativas para funções públicas e módulos.
- Adicione testes unitários para novas funções em `tests/`.
- Evite commitar pesos de modelos pesados (>50MB) ou imagens desnecessárias diretamente no Git.

---

## 🚀 Enviando seu Pull Request

1. Faça commit das alterações com mensagens claras e objetivas no padrão [Conventional Commits](https://www.conventionalcommits.org/) (ex.: `feat: ...`, `fix: ...`, `docs: ...`).
2. Envie suas alterações para o seu fork:
   ```bash
   git push origin sua-branch
   ```
3. Abra um **Pull Request (PR)** para a branch `main` do repositório original descrevendo as mudanças propostas e o impacto no projeto.
