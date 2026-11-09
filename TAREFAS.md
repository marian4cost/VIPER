# 📋 Lista de Tarefas - Reorganização do Projeto VIPER

- [x] **Etapa 1: Diagnóstico e Mapeamento Completo do Projeto**
  - [x] Mapear todo o histórico de commits do repositório Git (`marian4cost/VIPER`).
  - [x] Identificar a relação entre a pasta raiz (`viper`), o repositório oficial (`VIPER`) e a pasta de experimentos (`newTestes`).
  - [x] Catalogar todos os arquivos, pastas, códigos fontes, datasets e relatórios.
  - [x] Identificar com precisão o que é **realmente inútil e descartável** (venvs duplicados de ~22 GB, zips redundantes, cache pyc, git vazio na pasta pai).

- [x] **Etapa 2: Limpeza Segura de Arquivos e Pastas Inúteis**
  - [x] Remover os ambientes virtuais duplicados em `newTestes/*/venv` (liberando ~22 GB de espaço em disco).
  - [x] Remover zips redundantes (`novos_codigos.zip` e `dataset.zip`) cujos arquivos já se encontram extraídos.
  - [x] Remover caches `__pycache__` rastreados e arquivos duplicados de imagens em `novos_codigos/images`.
  - [x] Limpar o `.git` vazio na raiz `/viper/` que causa aninhamento incorreto de repositórios.
  - [x] *Commit da etapa de limpeza*.

- [ ] **Etapa 3: Integração e Reorganização Estrutural**
  - [ ] Integrar a pasta de experimentos/testes (`newTestes`) para dentro de `VIPER/experimentos/`.
  - [ ] Unificar os códigos-fonte: promover a versão modular e atualizada (`novos_codigos/` com telemetria e caminhos robustos) como o núcleo de código (`src/` e entrypoints).
  - [ ] Organizar os relatórios: consolidar todos os relatórios (`relatorio_*.txt`, `.pdf` e pasta legada) em `relatorios/`.
  - [ ] Manter pastas de dados, entradas e saídas de testes (`fotos_carros/`, `placas_processadas/`, `placas_finalizadas/`, `debug_saida_viper/`, `images/`).
  - [ ] *Commit da reorganização estrutural*.

- [ ] **Etapa 4: Atualização de Configurações, Dependências e Documentação**
  - [ ] Atualizar `.gitignore` de forma completa e profissional.
  - [ ] Ajustar `requeriments.txt` (ou `requirements.txt`).
  - [ ] Restaurar e aprimorar o `README.md` com a arquitetura completa, telemetria e instruções de uso.
  - [ ] *Commit da documentação e configurações*.

- [ ] **Etapa 5: Validação da Integridade e Testes de Importação**
  - [ ] Verificar a sintaxe e importações de todos os scripts Python reorganizados.
  - [ ] Testar execução dos módulos principais e de telemetria.
  - [ ] *Commit final de ajustes e validação*.

- [ ] **Etapa 6: Documentação e Explicação Detalhada Final**
  - [ ] Gerar documentação explicativa de cada pasta e arquivo do projeto para o usuário.
