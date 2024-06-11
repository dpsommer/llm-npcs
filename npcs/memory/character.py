from dataclasses import asdict, dataclass

import yaml

from npcs.utils import CHARACTER_CONFIG_PATH


def normalize(name: str) -> str:
    return name.lower().replace(" ", "_")


@dataclass
class Character:
    name: str
    age: int
    race: str
    gender: str
    location: str
    backstory: str

    def __str__(self) -> str:
        return f"""
Name: {self.name}
Age: {self.age}
Race: {self.race}
Gender: {self.gender}
Location: {self.location}
Backstory: {self.backstory}
"""

    @staticmethod
    def load_character(name: str) -> "Character":
        name = normalize(name)
        conf = CHARACTER_CONFIG_PATH / f"{name}.yaml"
        try:
            with open(conf, "r") as f:
                char_info = yaml.safe_load(f.read())
                return Character(**char_info)
        except Exception as e:
            # TODO: log errors
            print(e)

    def save(self) -> None:
        conf = CHARACTER_CONFIG_PATH / f"{normalize(self.name)}.yaml"
        with open(conf, "w") as f:
            f.write(yaml.dump(asdict(self)))
