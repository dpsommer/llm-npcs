.PHONY: init install test coverage

init: install test

install:
	pip install -r requirements-dev.txt
	python -m textblob.download_corpora
	python -m coreferee install en

test:
	pip install -qq --upgrade tox
	tox -p

coverage:
	pytest --cov-config .coveragerc --verbose --cov-report term --cov-report xml --cov=npcs tests
