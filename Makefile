.PHONY: init install trunk test coverage

TRUNK_INSTALLED=$(shell command -v trunk >/dev/null 2>&1 && echo 1 || echo 0)

init: install trunk test

install:
	@pip install -r requirements-dev.txt
	@python -m textblob.download_corpora
ifeq ($(TRUNK_INSTALLED), 0)
	@curl https://get.trunk.io -fsSL | bash -s -- -y
endif

trunk:
	@trunk fmt --all

test:
	@pip install -qq --upgrade tox
	@tox -p

coverage:
	@pytest --cov-config .coveragerc --verbose --cov-report term --cov-report xml --cov=npcs tests
