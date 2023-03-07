.ONESHELL:

include .env

VENV = venv
PYTHON = $(VENV)/Scripts/python
REPO_URL = https://github.com/Borg93/riks_ds_utils
PACKAGE = riks_ds_utils
PIP = $(VENV)/Scripts/pip
ACTIVATE_ENV = source venv/Scripts/activate
#windows :  .\venv\Scripts\activate #./venv/Scripts/activate
# linux, mac : source venv/Scripts/activate

help:
	@echo make connect_to_repo   - Connects local repo to github repo with the same name as the repo locally
	@echo make release           - provide tag and later push to origin
	@echo _
	@echo make venv              - Creates a venv, however, needs to be activated before use 
	@echo                          #windows: .\venv\Scripts\activate #./venv/Scripts/activate #linux mac : source venv/Scripts/activate
	@echo _
	@echo make local_install     - Installs python packages
	@echo make local_dev_install - Installs dev package to test code
	@echo _
	@echo make pre_commit        - Precommit hooks before git pushes, to format code before push 
	@echo _
	@echo make tests-unit        - Runs unit test and make tests-integration for you integration tests..
	@echo make lint              - Runs mypy and flake8
	@echo make test_tox          - Runs make lint and make test
	@echo _
	@echo make build             - Build python package
	@echo _
	@echo make clean_venv        - Removes Local venv 
	@echo make local_clean       - Removes Local folders such as .pytest_cache and other cache files..  
	@echo make local_build       - Combines make build and local_clean
	@echo _
	@echo make docker_build		 - docker build locally and run container
	@echo _
	@echo make docs_install      - install requirements for mkdocs

connect_to_repo:
	git init .
	git add --all
	git commit -m "First commit from CookieCutter"
	git branch -M main
	git remote add origin $(REPO_URL)
	git push -u origin main

# Make release
release:
	@echo git tag -a "vX.Y.Z" -m "Release X.Y.Z"
	@echo git push origin vX.Y.Z
	@echo change release vX.Y.Z in setup.py..

# Run only once..
pre_commit:
	$(PIP) install pre-commit
	pre-commit --version
	pre-commit install
	pre-commit run

# For local dev..
.PHONY: venv
venv:
	python -m venv $(VENV)
	$(ACTIVATE_ENV)

local_install:
	@echo _
	@echo Installing python packages...
	$(PIP) install -e .

test_install:
	@echo _
	@echo Installing python packages...
	pip install -e .
	pip install -r requirements_dev.txt

# For tests
local_dev_install:
	$(PIP) install -r requirements_dev.txt

.PHONY: test
tests-unit: local_dev_install
	pytest tests/unit -v

mock-tests-unit:
	pytest tests/unit -v

tests-integration:
	pytest test/integration

lint:
	mypy src
	flake8 src

test_tox:
	tox

# To build
local_build: build local_clean
	
build:
	@echo "Building package"
	pip install build
	python -m build

# Clean local dev
clean_venv: local_clean
	@echo "Cleaning previous python virtual environment..."
	deactivate
	rm -rf $(VENV)

.PHONY: local_clean
local_clean:
	@echo "Cleaning local folders:"
	if exist "dist" rmdir /S dist
	if exist ".pytest_cache" rmdir /S .pytest_cache
	if exist ".mypy_cache" rmdir /S .mypy_cache
	if exist ".tox" rmdir /S .tox
	if exist "src\package_name.egg-info" rmdir /S src\package_name.egg-info
	if exist "src\${PACKAGE}\__pycache__" rmdir /S src\${PACKAGE}\__pycache__
	if exist "tests\unit\__pycache__" rmdir /S tests\unit\__pycache__
	if exist ".coverage" del .coverage


docker_build:
	docker build -t riks_ds_utils .
	docker run riks_ds_utils
	@echo dont forget to remove image or rebuild it..

docs_install:
	pip install build mkdocs == 1.4.2
	pip install build mkdocs-material == 8.5.11
	pip install build mkdocs-jupyter == 0.22.0

