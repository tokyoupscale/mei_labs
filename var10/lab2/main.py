from utils.issuess import real_estate_issues

# Область применения - косметика, ее подбор на основе типа кожи и инд особенностей

# Возможные проблемы, сценарии и ситуации и связанные с ними хар-ки и симптомы

class RealEstateExpertSystem:
    def __init__(self):
        self.estate_issues = real_estate_issues

    def get_recommendations(self, market_type):
        if market_type in self.estate_issues:
            issue = self.estate_issues[market_type]
            print(f"Тип запроса: {market_type}")
            print(f"Описание: {issue['description']}\n")
            print("Рекомендации:")
            for rec in issue['recommendations']:
                print(f"- {rec}")
            print("Симптомы:")
            for sym in issue['symptoms']:
                print(f"- {sym}")
            
        else:
            print("неверный выбор\n")

choice = int(input("Введите тип запроса(1-5; \n1 - дом для семьи, \n2 - инвестирование в недвигу, \n3 - аренда, \n4 - нужен ремонт, \n5 - коммерческая недвига): \n"))

market_types = {
    1: "family_home_purchase",
    2: "investment_property",
    3: "rental_property",
    4: "property_requiring_renovation",
    5: "commercial_property"
}        

# skin_type = input("Введите тип кожи (1-5; \n1 - dry_skin, \n2 - oily_skin, \n3 - combination_skin, \n4 - sensitive_skin, \n5 - ageing_skin): ").lower()
skin_type = market_types.get(choice, None)
expert = RealEstateExpertSystem()
expert.get_recommendations(skin_type)