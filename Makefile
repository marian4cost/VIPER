.PHONY: help install install-dev run test report train validate clean

help:
	@echo "Comandos disponíveis no VIPER:"
	@echo "  make install      - Instala dependências principais do projeto"
	@echo "  make install-dev  - Instala dependências de teste e formatação"
	@echo "  make run          - Executa o pipeline integrado principal (main.py)"
	@echo "  make test         - Executa os testes unitários com pytest"
	@echo "  make report       - Gera o relatório técnico em PDF (gerar_pdf_relatorio.py)"
	@echo "  make train        - Inicia o treinamento do modelo YOLO de placas"
	@echo "  make validate     - Executa benchmark e validação de métricas/latência"
	@echo "  make clean        - Remove caches Python e arquivos temporários"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt -r requirements-dev.txt

run:
	python main.py

test:
	pytest -v

report:
	python gerar_pdf_relatorio.py

train:
	python trainYOLO.py

validate:
	python validadeYOLO.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
