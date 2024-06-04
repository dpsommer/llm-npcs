import os
import shutil
from pathlib import Path

import pytest

from npcs.memory import NLPPipeline
from npcs.memory.schema import NPCMemory
from npcs.memory.search import add_memories, load_index, ram_index

TEST_DIR_BASEPATH = Path(__file__).resolve().parent
TEST_INDEX_DIR = TEST_DIR_BASEPATH / "test_index"


def pytest_addoption(parser):
    parser.addoption(
        "--functional",
        action="store_true",
        help="Functional flag to run integration tests",
    )


def pytest_runtest_setup(item):
    if "functional" in item.keywords and not item.config.getoption("--functional"):
        pytest.skip("Use the --functional flag to run functional tests")


def pytest_configure():
    pytest.TEST_DIR_BASEPATH = TEST_DIR_BASEPATH
    pytest.DATA_DIR = TEST_DIR_BASEPATH / "data"


# Global fixtures
@pytest.fixture
def clean_test_subtree():
    def clean_tree(dir: Path):
        if dir.is_dir():
            try:
                shutil.rmtree(dir)
            except Exception as e:
                print(f"Failed to delete {dir}: {e}")
            os.rmdir(dir)

    return clean_tree


@pytest.fixture(scope="package")
def nlp():
    return NLPPipeline()


@pytest.fixture
def memories():
    return [
        NPCMemory(
            npc="John Doe",
            memory="A mug of ale costs 4 copper pieces.",
            entities={"ale"},
            sentiment_polarity=0.0,
        ),
        NPCMemory(
            npc="John Doe",
            memory="This inn's name is the Silver Fox.",
            entities={"Silver Fox"},
            sentiment_polarity=0.0,
        ),
        NPCMemory(
            npc="John Doe",
            memory="It costs 1 silver piece a night to stay at the inn.",
            entities={"Silver Fox"},
            sentiment_polarity=0.0,
        ),
        NPCMemory(
            npc="John Doe",
            memory="A mug of ale costs 4 copper pieces at the Silver Fox.",
            entities={"ale", "Silver Fox"},
            sentiment_polarity=0.0,
        ),
    ]


@pytest.fixture
def in_memory_index(memories):
    index = ram_index()
    add_memories(index=index, data=memories)
    yield index
    index.close()


@pytest.mark.functional
@pytest.fixture
def file_based_index(memories, clean_test_subtree):
    test_index = load_index(TEST_INDEX_DIR, "test")
    add_memories(index=test_index, data=memories)
    yield test_index
    clean_test_subtree(TEST_INDEX_DIR)
