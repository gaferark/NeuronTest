def main():
    with open('Equations.txt', 'w', encoding='utf-8') as file:
        for b in range(-100, 101):
            for c in range(-100, 101):
                bool_to_int = {True: 1, False: 0}
                file.write(f"{b} {c} {bool_to_int[not ((b ** 2 - 4 * c) < 0)]}\n")


def normalize(x):
    return (x + 50) / 2550


if __name__ == '__main__':
    main()
