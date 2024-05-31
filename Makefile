.PHONY: init build test coverage

init: build test

build:
	pip install -r requirements-dev.txt

test:
	pip install -qq --upgrade tox
	tox -p

coverage:
	pytest --cov-config .coveragerc --verbose --cov-report term --cov-report xml --cov=npcs tests
