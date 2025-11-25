from typing import Union


class Ask:
    def __init__(self, choices=['y', 'n']):
        self.choices = choices

    def ask(self):
        if max(len(x) for x in self.choices) > 1:
            for i, x in enumerate(self.choices):
                print(f"{i}. {x}")
            while True:
                try:
                    idx = int(input("номер варианта: "))
                    if 0 <= idx < len(self.choices):
                        return self.choices[idx]
                except ValueError:
                    pass
        else:
            print("/".join(self.choices))
            ans = input().strip().lower()
            return ans or self.choices[0]


class Content:
    def __init__(self, x):
        self.x = x

class If(Content):
    pass

class AND(Content):
    pass

class OR(Content):
    pass

Rule = Union[Ask, If, AND, OR, list[str]]

class KnowledgeBase:
    def __init__(self, rules: dict[str, Rule]):
        self.rules = rules
        self.memory = {}

    def get(self, name):
        if name in self.memory:
            return self.memory[name]

        for fld in self.rules.keys():
            if fld == name or fld.startswith(name + ":"):
                value = 'y' if fld == name else fld.split(':', 1)[1]
                res = self.eval(self.rules[fld], field=name)
                if res == 'y':
                    self.memory[name] = value
                    return value

        res = self.eval(self.rules['default'], field=name)
        self.memory[name] = res
        return res

    def eval(self, expr, field=None):
        if isinstance(expr, Ask):

            if isinstance(field, str):
                label = field.replace('_', ' ')
            else:
                label = "уточните"

            print(f"\n{label}?")
            return expr.ask()

        elif isinstance(expr, If):
            return self.eval(expr.x, field=field)

        elif isinstance(expr, AND) or isinstance(expr, list):
            vals = expr.x if isinstance(expr, AND) else expr
            for x in vals:
                if self.eval(x) == 'n':
                    return 'n'
            return 'y'

        elif isinstance(expr, OR):
            for x in expr.x:
                if self.eval(x) == 'y':
                    return 'y'
            return 'n'

        elif isinstance(expr, str):
            return self.get(expr)

        else:
            return 'n'

# будут спрашиваться по default
questions = [
    'считаете_кожу_сухой',
    'считаете_кожу_жирной',
    'считаете_кожу_комбинированной',
    'считаете_кожу_нормальной',
    'есть_стянутость_после_умывания',
    'есть_шелушения',
    'есть_жирный_блеск',
    'есть_частые_прыщи_или_воспаления',
    'кожа_часто_краснеет_или_реагирует',
    'есть_тенденция_к_морщинам',
    'есть_пигментация_или_пятна',
    'нравятся_легкие_текстуры',
]

rules: dict[str, Rule] = {
    'default': Ask(['y', 'n']),
    'сухая_кожа': If(OR([
        'считаете_кожу_сухой',
        'есть_стянутость_после_умывания',
        'есть_шелушения',
    ])),

    'жирная_кожа': If(OR([
        'считаете_кожу_жирной',
        'есть_жирный_блеск',
    ])),

    'комбинированная_кожа': If(OR([
        'считаете_кожу_комбинированной',
        AND(['есть_жирный_блеск', 'есть_стянутость_после_умывания']),
    ])),

    'нормальная_кожа': If('считаете_кожу_нормальной'),

    'чувствительная_кожа': If(OR([
        'кожа_часто_краснеет_или_реагирует',
        'есть_аллергия_на_отдушки',
    ])),

    'склонность_к_акне': If('есть_частые_прыщи_или_воспаления'),

    'нужен_антиэйдж': If('есть_тенденция_к_морщинам'),
    'нужна_коррекция_пигментации': If('есть_пигментация_или_пятна'),

    'профиль:сухая_чувствительная': If(AND([
        'сухая_кожа',
        'чувствительная_кожа',
    ])),

    'профиль:жирная_акне': If(AND([
        'жирная_кожа',
        'склонность_к_акне',
    ])),

    'профиль:комбинированная_чувствительная': If(AND([
        'комбинированная_кожа',
        'чувствительная_кожа',
    ])),

    'профиль:нормальная': If('нормальная_кожа'),

    'профиль:жирная_без_акне': If(AND([
        'жирная_кожа',
    ])),
}

recommendations = {
    'сухая_чувствительная': '\nумывание: мягкое средство без спирта (крем-гель или молочко), \nтоник: без спирта, с успокаивающими компонентами, \nкрем: плотный крем с церамидами, маслами, гиалуроновой кислотой \nSPF: минеральный или гибридный фильтр.',
    'жирная_акне': '\nумывание: мягкий гель без агрессивных сульфатов, 2 раза в день, \nтоник/сыворотка: ниацинамид, BHA., \nкрем: лёгкий флюид/гель, без тяжёлых масел, \nSPF: лёгкий матирующий флюид, некомедогенный.',
    'комбинированная_чувствительная': '\nумывание: мягкий гель или пенка, не пересушивающие, \nкрем: лёгкий без спирта, с успокаивающими компонентами, \nSPF: лёгкий крем/флюид без агрессивных отдушек.',
    'нормальная': '\nумывание: мягкий гель/пенка по потребности, \nкрем: базовый увлажняющий крем с глицерином/гиалуроновой кислотой, \nSPF: ежедневный SPF 30–50 с комфортной текстурой.,.',
    'жирная_без_акне': '\nумывание: мягкий гель, \nтоник/сыворотка: ниацинамид, лёгкие кислоты по переносимости, \nкрем: лёгкий увлажняющий гель, \nSPF: матирующий или лёгкий флюид..'
}


def main():
    kb = KnowledgeBase(rules)
    profile = kb.get('профиль')
    
    print(f"определённый профиль кожи: {profile}")

    rec = recommendations.get(profile)
    if rec:
        print(rec)
    else:
        print("профиль не удалось определить однозначно. "
            "рекомендуется базовый мягкий уход и универсальный SPF.")


if __name__ == '__main__':
    main()
