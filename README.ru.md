[![en](https://img.shields.io/badge/lang-EN-blue.svg)](https://github.com/CompPhysLab/SVETlANNa/blob/main/README.md)
[![ru](https://img.shields.io/badge/lang-RU-green.svg)](https://github.com/CompPhysLab/SVETlANNa/blob/main/README.ru.md)

# SVETlANNa

SVETlANNa — это библиотека на языке Python с открытым исходным кодом для моделирования распространения оптических пучков в оптических схемах в свободном пространстве и нейроморфных систем, таких как дифракционные нейронные сети. Библиотека основана на фреймворке PyTorch, используя его ключевые возможности, такие как тензорные вычисления и эффективная параллельная обработка данных. В своей основе SVETlANNa опирается на Фурье-оптику, поддерживая несколько моделей распространения, включая метод углового спектра и приближение Френеля.

Сопровождающий проект на github [SVETlANNa.docs](https://github.com/CompPhysLab/SVETlANNa.docs) содержит различные примеры использования библиотеки в формате блокнотов Jupyter.

Название библиотеки образовано от русского слова «свет» и аббревиатуры ANN, что означает искусственная нейронная сеть, и одновременно все слово звучит как русское женское имя Светлана.

## Сокращения

НС - нейронная сеть

ИНС - искусственная нейронная сеть

ОНС - оптическая нейронная сеть

ДОНС - дифракционная оптическая нейронная сеть

ДОЭ - дифракционный оптический элемент

SLM - пространственный модулятор света

## Особенности библиотеки

- основана на [PyTorch](https://pytorch.org/)
- модели прямого распространения  the Angular spectrum method and the Fresnel approximation
- possibility to solve the classical DOE/SLM optimization problem with the Gerchberg-Saxton and hybrid input-output algorithms
- support for custom elements and optimization methods
- support for various free-space ONN architectures including feed-forward NN, autoencoders, and recurrent NN
- cross platform
- full GPU aceleration
- companion repository with numerous .ipynb examples
- custom logging, project management, analysis, and visualization tools
- tests for the whole functionality

## Перечень направлений прикладного использования

- моделирование и оптимизация оптических систем и оптических пучков, распространяющихся в свободном пространстве;
- расчет параметров фазовых масок, ДОЭ и SLM как для классических оптических систем, так и для нейроморфных оптических вычислителей;
- моделирование и оптимизация параметров ОНС и ДОНС для задач полностью оптической классификации и прогнозирования.

## Рекомендуемые минимальные технические требования для запуска и использования библиотеки:

1.	процессор Intel Core i5 8400 / AMD Ryzen 5 1400 или выше;
2.	оперативная память не менее 8 Гб;
3.	свободного места на диске не менее 16 Гб (виртуальное окружение занимает около 6 Гб);
4.	при запуске расчетов на видеокарте требуется карта Nvidia с поддержкой CUDA и памятью не менее 2 Гб;
5.	операционная система Windows версии не ниже 10 или Linux с версией ядра не ниже 5.15;
6.	библиотека работает на версии Python 3.11.

# Установка, использование и примеры

Чтобы использовать библиотеку есть две опции - установить исходники или установка через pip.

## Установка из PyPI

```bash
pip install svetlanna
```

## Установка исходных файлов

Перед установкой необходимо установить [git](https://git-scm.com).

1. Клонируйте репозиторий:
```bash
  git clone https://github.com/CompPhysLab/SVETlANNa.git
```
2. Перейдите в папку библиотеки:
```bash
  cd SVETlANNa
```
3. Создайте виртуальное окружение для Python 3.11 и активируйте его (см. документацию [venv](https://docs.python.org/3/library/venv.html)). Все последующие действия необходимо выполнять внутри активированного окружения.
4. Установите Poetry (версии 2.0.0 или выше):
```bash
  pip install poetry
```
5. Установите зависимости:
```bash
  poetry install --all-extras
```
6. Установите PyTorch (см. инструкции на официальном [сайте](https://pytorch.org/get-started/locally/)). Выберите подходящий вариант установки в зависимости от вашей системы. Для работы с GPU требуется соответствующая видеокарта и поддерживаемая сборка PyTorch.
```bash
  pip install torch
```

## Запуск тестов

Тесты могут быть запущены только при установке библиотеки из исходных файлов. Версия, устанавливаемая из PyPI не включает тесты.

ВНИМАНИЕ: запуск тестов требует, как минимум, 4 Гб оперативной памяти.

Для запуска тестов выполните следующую команду (в папке библиотеки при активированном виртуальном окружении - см. выше)
```bash
  pytest
```

## Документация

Документация доступна по [ссылке](https://compphyslab.github.io/SVETlANNa/)

## Запуск примеров с установкой основной библиотеки

Примеры с различными сценариями использования библиотеки собраны в репозитории [SVETlANNa.docs](https://github.com/CompPhysLab/SVETlANNa.docs).

Чтобы запустить примеры локально на вашем компьютере выполните следующие действия:
1. Клонируйте репозиторий (должен быть установлен [git](https://git-scm.com)):
```bash
  git clone https://github.com/CompPhysLab/SVETlANNa.docs.git
```
2. Перейдите в директорию проекта:
```bash
  cd SVETlANNa.docs
```
3. Создайте виртуальное окружение для Python 3.11 (см. документацию по [venv](https://docs.python.org/3/library/venv.html)) и активируйте его. Все последующие действия необходимо выполнять внутри активированного окружения.
4. Установите основную библиотеку:
```bash
  pip install svetlanna
```
5. Установите актуальную версию PyTorch (см. инструкции на официальном [сайте](https://pytorch.org/get-started/locally/)). Выберите подходящий вариант установки в зависимости от вашей системы. Для работы с GPU требуется соответствующая видеокарта и поддерживаемая сборка PyTorch.
6. Если в системе не установлен Jupyter notebook, установите его:
```bash
  pip install notebook
```
7. Установите дополнительные пакеты:
```bash
  pip install reservoirpy matplotlib tqdm requests av scikit-image py-cpuinfo gputil
```
8. Запустите Jupyter notebook и откройте страницу в бразере (см. “Or copy and paste one of these URLs:” в терминале):
```bash
  jupyter notebook
```
9. Перейдите в папку проекта SVETlANNa.docs. Примеры разделены по тематическим подпапкам в виде .ipynb файлов. Для запуска примера требуется открыть нужный файл в интерфейсе Jupyter notebook и выполнить все ячейки с кодом (комбинация клавиш Shift+Enter).

## Запуск примеров с установленной из исходников библиотекой SVETlANNa

1. Клонируйте репозиторий (должен быть установлен [git](https://git-scm.com)):
```bash
	git clone https://github.com/CompPhysLab/SVETlANNa.docs.git 
```
Дальнейшие команды выполняются при активированном venv, который ранее использовался при установке библиотеки SVETlANNa из исходников.
2. Если в системе не установлен Jupyter notebook, установите его:
```bash
  pip install notebook
```
3. Установите дополнительные пакеты
	pip install reservoirpy matplotlib tqdm requests av scikit-image py-cpuinfo gputil
4. Запустите Jupyter notebook:
```bash
  jupyter notebook
```
и откройте страницу в бразере (см. “Or copy and paste one of these URLs:” в терминале)
5. Перейдите в папку проекта SVETlANNa.docs. Примеры разделены по тематическим подпапкам в виде .ipynb файлов. Для запуска примера требуется открыть нужный файл в интерфейсе Jupyter notebook и выполнить все ячейки с кодом (комбинация клавиш Shift+Enter).

## Примеры

Ниже показан пример результата обучения оптической нейронной сети для задачи классификации MNIST: изображение цифры «8» пропускается через стек из 10 фазовых пластин с оптимизированными функциями пропускания. Выбранные области детектора соответствуют разным распознаваемым цифрам. Класс распознаваемой цифры определяется областью детектора, измеряющей максимальную оптическую интенсивность.

Примеры визуализации оптической установки и полей в срезе пучка:

<img src="./pics/visualization.png" alt="drawing" width="400"/>

Example a of five-layer DONN trained to recognize numbers from the MNIST database:

<img src="./pics/MNIST example 1.png" alt="drawing" width="400"/>

<img src="./pics/MNIST example 2.png" alt="drawing" width="400"/>

<img src="./pics/MNIST example 3.png" alt="drawing" width="400"/>

# Вклад в разработку

Улучшение библиотеки и разработка новых модулей приветствуются (см. файлы `contributing.md`).

# Бладарности

Работа над данным проектом была поддержана [Фондом содействия инновациям](https://en.fasie.ru/)

# Авторы

- [@aashcher](https://github.com/aashcher)
- [@alexeykokhanovskiy](https://github.com/alexeykokhanovskiy)
- [@Den4S](https://github.com/Den4S)
- [@djiboshin](https://github.com/djiboshin)
- [@Nevermind013](https://github.com/Nevermind013)

# Лицензия

[Mozilla Public License Version 2.0](https://www.mozilla.org/en-US/MPL/2.0/)
