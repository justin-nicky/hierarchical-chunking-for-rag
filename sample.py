class Entity():
    def __init__(self, damage_amt, health):
      self.damage_amt = damage_amt
      self.health = health

    def attack(self):
       self.health = self.health - self.damage_amt
       print(f'attack! Damage amount: {self.damage_amt}')


class Monster(Entity):
    def __init__(self, health, damage_amt):
      self.health = health
      self.damage_amt = damage_amt
      super().__init__(damage_amt, health)

    def __str__(self): 
       return f'A monster with {self.health}hp'

monster = Monster(80,5)
print(monster)
monster.attack()
print(monster)

    
