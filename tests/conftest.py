import os
import shutil

import pytest

TEST_DIR_BASEPATH = os.path.abspath(os.path.dirname(__file__))


def pytest_addoption(parser):
    parser.addoption('--functional', action='store_true', help="Functional flag to run integration tests")


def pytest_runtest_setup(item):
    if 'functional' in item.keywords and not item.config.getoption('--functional'):
        pytest.skip("Use the --functional flag to run functional tests")


def pytest_configure():
    pytest.TEST_DIR_BASEPATH = TEST_DIR_BASEPATH
    pytest.DATA_DIR = os.path.join(TEST_DIR_BASEPATH, 'data')


@pytest.fixture
def clean_test_subtree():
    def clean_tree(dir):
        if os.path.isdir(dir):
            for filename in os.listdir(dir):  # teardown
                file_path = os.path.join(dir, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))
            os.rmdir(dir)
    return clean_tree
