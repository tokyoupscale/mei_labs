# Область применения - косметика, ее подбор на основе типа кожи и инд особенностей

# Возможные проблемы, сценарии и ситуации и связанные с ними хар-ки и симптомы
from utils.issues import cosmetic_system_issues

class CosmeticExpertSystem:
    def __init__(self):
        self.cosmetic_issues = cosmetic_system_issues

    def get_recommendations(self, skin_type):
        if skin_type in self.cosmetic_issues:
            issue = self.cosmetic_issues[skin_type]
            print(f"Тип кожи: {skin_type}")
            print(f"Описание: {issue['description']}\n")
            print("Рекомендации:")
            for rec in issue['recommendations']:
                print(f"- {rec}")
            print("Симптомы:")
            for sym in issue['symptoms']:
                print(f"- {sym}")
            
        else:
            print("неверный тип кожи\n")

choice = int(input("Введите тип кожи (1-5; \n1 - dry_skin, \n2 - oily_skin, \n3 - combination_skin, \n4 - sensitive_skin, \n5 - ageing_skin): \n"))

skin_types = {
    1: "dry_skin",
    2: "oily_skin",
    3: "combination_skin",
    4: "sensitive_skin",
    5: "ageing_skin"
}        

# skin_type = input("Введите тип кожи (1-5; \n1 - dry_skin, \n2 - oily_skin, \n3 - combination_skin, \n4 - sensitive_skin, \n5 - ageing_skin): ").lower()
skin_type = skin_types.get(choice, None)
expert = CosmeticExpertSystem()
expert.get_recommendations(skin_type)