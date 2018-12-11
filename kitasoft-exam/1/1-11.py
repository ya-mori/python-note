
if __name__ == '__main__':

    x = 10
    y = 8

    if not x > y:
        raise Exception('x が y より小さいです。')

    product = int(x / y)
    remainder = x % y

    print(f"x / y = {product} ・・・ {remainder}")
