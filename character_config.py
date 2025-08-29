# character_config.py
import yaml
import logging
import os

class CharacterConfig:
    def __init__(self, filename="character_config.yaml"):
        self.filename = filename
        self.characters = {}
        self.reload_config()

    def reload_config(self):
        if not os.path.exists(self.filename):
            logging.warning(f"找不到 {self.filename}")
            self.characters = {}
            return False
        try:
            with open(self.filename, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                self.characters = data.get("characters", {})
            logging.info("角色設定已重新載入")
            return True
        except Exception as e:
            logging.error(f"載入角色設定失敗: {e}")
            return False

    def build_character_prompt(self, name, facts=[]):
        char = self.characters.get(name)
        if not char:
            char = self.characters.get("希雅", {"description": "你是希雅，一個情緒穩定的人", "facts": []})
        description = char.get("description", "")
        char_facts = char.get("facts", [])
        persona = description
        if char_facts:
            persona += "\n角色基礎資料：" + "；".join([str(f) for f in char_facts])
        if facts:
            persona += "\n已知對方資料：" + "；".join(facts)
        return persona
