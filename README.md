# Статья об использовании `std::function`

Поводом для написания данной статьи стало желание систематизировать всё то, чему я научился в процессе кодинга лаб по таким предметам, как Численные методы; Уравнения математической физики; Методы оптимизации. Надеюсь эта статья будет полезна не только людям, которые учатся в НГТУ на факультете ФПМИ, но и широкому кругу читателей.

Все примеры для данной статьи взяты из кода лабораторных по следующим предметам: [ЧМ](https://github.com/optozorax/numerical_methods), [УМФ](https://github.com/optozorax/labs_emf), [МО](https://github.com/optozorax/optimization_methods).

# Введение

В Си/C++ имеется возможность передавать функцию в качестве аргумента функции как указатель на функцию. Вы знаете этот страшный синтаксис:

```c++
int add(int x, int y) {
	return x+y;
}

int (*operation)(int, int) = add;

int c = operation(1, 2);
```

Начиная с C++11 появился удобный интерфейс для описания функциональных объектов под названием `std::function`, он имеет более красивый синтаксис:

```c++
#include <functional>

int add(int x, int y) {
	return x+y;
}

std::function<int(int, int)> operation = add;

int c = operation(1, 2); // c = 3
```

Причём, в отличие от Си, в C++ таким образом можно описывать любой функциональный объект, то есть объект, допускающий вызов операции `()`:

```c++
#include <functional>

struct Adder
{
	int operator()(int a, int b) {
		return state + a + b;
	}
	
	int state;
};

Adder add = {1};

std::function<int(int, int)> operation = add;

int c = operation(1, 2); // c = 4
```

Аналогичным образом можно использовать [лямбды](https://habr.com/ru/post/66021/):

```c++
#include <functional>

int state = 1;

std::function<int(int, int)> operation = [&](int a, int b) -> int {
	return state + a + b;
};

int c = operation(1, 2); // c = 4
```

Таким образом, можно передавать функции в качестве аргументов функций.

# Численное вычисление производной

Пусть у нас имеется некоторая функция `f`, которая получает `double` и возвращает `double`. Тогда можно вычислить производную этой функции следующим образом:

```c++
double f(double x);

double derivative_f(double x) {
	double h = 0.0001;
	return (f(x+h)-f(x))/h;
}
```

Но писать для каждой функции другую функцию, вычисляющую её производную непрактично! Поэтому такую проблему можно решить с помощью `std::function`:

```c++
std::function<double(double)> derivative(std::function<double(double)> f) {
	return [f] (double x) -> double {
		double h = 0.0001;
		return (f(x+h)-f(x))/h;
	}
}
```

Данная функция получает функцию, и возвращает функцию, которая считает её производную. Пример использования:

```c++
double f(double x) {
	return x*x;
}

auto df = derivative(f);

double a = f(1); // a == 1
double a = df(1); // a == 2
```

А далее представлен код для вычисления первой и второй производных функций, взятый из [курсовой по УМФ](https://github.com/optozorax/labs_emf/blob/60b62fb5746aa83e3c52a802d5da1741950b6292/coursework/fem.cpp#L283), смело используйте его в своих проектах.

```c++
typedef std::function<double(double)> function_1d_t; 

function_1d_t calc_first_derivative(const function_1d_t& f) {
	return [f](double x) -> double {
		const double h = 0.001;
		return (-f(x + 2 * h) + 8 * f(x + h) - 8 * f(x - h) + f(x - 2 * h)) / (12 * h);
	};
}

function_1d_t calc_second_derivative(const function_1d_t& f) {
	return [f](double x) -> double {
		const double h = 0.001;
		return (-f(x+2*h) + 16*f(x+h) - 30*f(x) + 16*f(x-h) - f(x-2*h))/(12*h*h);
	};
}
```

Может показаться, что если уменьшить величину `0.001`, то точность повысится, но это не так, на практике у меня точность только падала, а при этом значении уже очень хорошо расчитывается производная.


# Пример использования `std::bind`

Окей, у нас существует функция для расчета производной одномерной функции, но что если у нас функция принимает два аргумента, например `f(x, y)`. Как посчитать её производную по одному параметру? 

На помощь приходит `std::bind`, и лучше увидеть пример, чтобы понять как он работает:

```c++
double f(double x, double y) {
	return 2*x + 3*y;
}

// Производная f по x при y=0
auto fx = calc_first_derivative(std::bind(f, std::placeholders::_1, 0));

// Производная f по y при x=0
auto fy = calc_first_derivative(std::bind(f, 0, std::placeholders::_1));

double fx_value = fx(1); // fx_value = 2; 
double fy_value = fy(1); // fy_value = 3;
```

К сожалению в этом случае функция `fx` получается функцией одной переменной, при фиксированном `y=0`. Чтобы получить функцию `fx`, которая принимает два параметра, но при этом возвращает производную по `x`, можно воспользоваться следующим трюком:

```c++
auto fx = [] (double x, double y) -> double {
	return calc_first_derivative(std::bind(f, std::placeholders::_1, y))(x);
};

double fx_value = fx(1, 0); // df(1, 0)/dx
```

Или другой пример:

```c++
double f(double x, double y, double t);

using namespace std::placeholders;

std::function<double(double, double)> g = std::bind(f, _2, 5, _1);

double gv = g(1, 2);
double fv = f(2, 5, 1);

// gv == fv
```

# Автоматический расчет правой части

В таком предмете, как УМФ требуется решить следующее дифференциальное уравнение: `-div(lambda * grad u) + gamma * u + sigma * du/dt = f`, где `u = u(x, y, t)` является неизвестной функцией. Для численного решения используется Метод Конечных Элементов.

Для декартовых координат это уравнение раскладывается в `lambda*(d^2 u/dx^2 + d^2 u/dy^2)  + gamma * u + sigma * du/dt = f`

Для тестирования программы-решателя мы придумываем некоторую функцию `u`, например: `u = x*x + y*y + t`, придумываем значения констант `lambda`, `gamma`, `sigma`, и согласно верхнему уравнению можно вычислить функцию `f`, при которой дифференциальное уравнение превращается в тождество.

Для того, чтобы тестировать нашу программу на широком спектре функций, можно написать функцию, которая будет автоматически рассчитывать эту правую часть на основе вышеописанных функций для вычисления производных:

```c++
typedef std::function<double(double, double, double)> function_3d_t;

/** Все константы решаемого уравнения. */
struct constants_t
{
	double lambda; /// Коэффициент внутри div
	double gamma;  /// Коэффициент при u
	double sigma;  /// Коэффициент при du/dt
};

function_3d_t calc_right_part(
	const function_3d_t& u,
	const constants_t& cs
) {
	// f = -div(lambda * grad u) + gamma * u + sigma * du/dt
	return [=](double x, double y, double t) -> double {
		using namespace placeholders;
		auto ut = calc_first_derivative(bind(u, x, y, _1));

		auto uxx = calc_second_derivative(bind(u, _1, y, t));
		auto uyy = calc_second_derivative(bind(u, x, _1, t));

		return -cs.lambda * (uxx(x) + uyy(y)) + cs.gamma * u(x, y, t) + cs.sigma * ut(t);
	};
}
```

Код взят опять же из [репозитория по УМФ](https://github.com/optozorax/labs_emf/blob/60b62fb5746aa83e3c52a802d5da1741950b6292/coursework/fem.cpp#L304).

Таким образом, мы получаем функцию `f` на основе известной нам функции `u`.

Это избавляет от лишней рутины ручного вычисления производных для функций при тестировании, автоматизируя этот процесс, также снижается вероятность ошибиться.

# Обертка для подсчета вызовов функции

Например, нам надо протестировать какой-то метод, находящий минимум заданной функции `f`. Но надо протестировать его на количество вызовов функции `f`. Это можно сделать не вмешиваясь в код этого метода, явным образом выставляя `count++` при каждом вызове функции, а можно поступить более красиво, послав вместо функции обёртку над ней:

```c++
typedef std::function<double(const Vector&)> Function;

Function setFunctionToCountCalls(int* where, const Function& f) {
	(*where) = 0;
	return [where, f](const Vector& v) -> double {
		(*where)++;
		return f(v);
	};
}

double f(const Vector& v) {
	// ...
}

int fCount = 0;
auto result = optimize(/* ... */, setFunctionToCountCalls(&fCount, f), /* ... */);
```

Взято из [2 лабораторной по МО](https://github.com/optozorax/optimization_methods/blob/feb65cda80b43145a58ada90b6de7c5ae8e777a4/2/methods.cpp#L268).


# Замер времени

У нас стоит задача замерить время работы какого-то кода. Можно поступить следующим образом:

```c++
#include <chrono>

using namespace chrono;
auto start = high_resolution_clock::now();

// ...
// main code
// ...

auto end = high_resolution_clock::now();
double time = duration_cast<microseconds>(end - start).count();
```

Но это решение плохо тем, что нам постоянно надо копировать эти участки кода, и мы потенциально можем потерять начало или конец замера времени, поэтому можно воспользоваться концепцией RAII, и сделать это следующим красивым образом:

```c++
#include <chrono>
#include <functional>

/** Считает время выполнения функции f в микросекундах. */
inline double calc_time_microseconds(const std::function<void(void)>& f) {
	using namespace std::chrono;
	auto start = high_resolution_clock::now();
	f();
	auto end = high_resolution_clock::now();
	return duration_cast<microseconds>(end - start).count();;
}

double time = calc_time_microseconds([&](){
	// ...
	// main code
	// ...
});
```

Притом мы не теряем локальные переменные благодаря использованию лямбд.

Код взят из [курсовой по УМФ](https://github.com/optozorax/labs_emf/blob/60b62fb5746aa83e3c52a802d5da1741950b6292/coursework/lib.h#L24).

# Двойной интеграл

Предположим, что у нас есть функция вычисления интеграла функции:

```c++
#include <functional>

typedef std::function<double(double)> function_1d_t; 

double calc_integral_gauss3(
	double a, double b, int n, // n - количество внутренных узлов
	const function_1d_t& f
);
```

Тогда двойной интеграл двумерной функции можно вычислить следующим образом:

```c++
double calc_integral_gauss3(
	double ax, double bx, int nx,
	double ay, double by, int ny,
	const function_2d_t& f
) {
	return calc_integral_gauss3(ax, bx, nx, [ay, by, ny, f](double x)->double {
		return calc_integral_gauss3(ay, by, ny, bind(f, x, _1));
	});
}
```

Код взят из [курсовой по УМФ](https://github.com/optozorax/labs_emf/blob/60b62fb5746aa83e3c52a802d5da1741950b6292/coursework/fem.cpp#L269).

# Функции тестирования

Представим, у нас есть множество методов, решающих одну задачу, но каждый по-разному, при этом интерфейс у них одинаковый. И стоит задача протестировать их все единым образом, построить таблицы. Это тоже можно красиво сделать с помощью `std::function`, написать прототип функции метода:

```c++
typedef std::function<MethodResult(/* Method args. */)> Method;

MethodResult method1(/* Method args. */);
MethodResult method2(/* Method args. */);
MethodResult method3(/* Method args. */);
MethodResult method4(/* Method args. */);
MethodResult method5(/* Method args. */);

void makeTableForMethod(
	const Method& method,
	const std::string& file,
	/* Method args. */	
) {
	// ...
}

makeTableForMethod(method1, "table1.txt", /* Method args. */);
makeTableForMethod(method2, "table1.txt", /* Method args. */);
makeTableForMethod(method3, "table1.txt", /* Method args. */);
makeTableForMethod(method4, "table1.txt", /* Method args. */);
makeTableForMethod(method5, "table1.txt", /* Method args. */);
```

Некоторые люди делают это с помощью копипаста, но думаю не стоит вам объяснять чем плох копипаст по сравнению с этим подходом.

Код взят из [2 лабораторной по МО](https://github.com/optozorax/optimization_methods/blob/feb65cda80b43145a58ada90b6de7c5ae8e777a4/2/make_tables.cpp#L9).

# Одномерный поиск

В МО многомерные методы нахождения минимума функции используют одномерную функцию оптимизации. И было задание задавать различные методы одномерной оптимизации, чтобы протестировать их эффективность конкретно в этой среде. Ну раз так, то мы не будем в программе жестко задавать функцию одномерной оптимизации, а будем передавать её как аргумент функции многомерной оптимизации:

```c++
typedef std::function<double(const double&)> OneDimensionFunction;
typedef std::function<double(const OneDimensionFunction&, double, double, double)> OneDimenshionExtremumFinder;

double oneDimensionOptimizator1(const OneDimensionFunction& f, double a, double b, double eps);
double oneDimensionOptimizator2(const OneDimensionFunction& f, double a, double b, double eps);
double oneDimensionOptimizator3(const OneDimensionFunction& f, double a, double b, double eps);
double oneDimensionOptimizator4(const OneDimensionFunction& f, double a, double b, double eps);


typedef std::function<double(const Vector&)> Function;

MethodResult multiDimenshionOptimize1(
	const Function& f, 
	const OneDimenshionExtremumFinder& argmin, 
	const Vector& x0, 
	const double& eps
);

MethodResult multiDimenshionOptimize2(
	const Function& f, 
	const OneDimenshionExtremumFinder& argmin, 
	const Vector& x0, 
	const double& eps
);
```

Код взят из [2 лабораторной по МО](https://github.com/optozorax/optimization_methods/blob/feb65cda80b43145a58ada90b6de7c5ae8e777a4/2/methods.h#L18).

# Паттерн listener

Иногда бывает так, что вам нужно выводить информацию о внутреннем состоянии метода, например, вы решаете задачу оптимизации, и вам надо на каждой итерации выводить текущее решение, значение градиента функции в этой точке итд. Вместо того, чтобы делать это явно, можно передавать в функцию метода функцию `listener`, которая будет получать всё внутреннее состояние метода, и уже у себя решать что с этим делать.

В одном случае она может выводить это на экран, в другом случае - на экран, а в третьем, когда нужная максимальная производительность - ничего не делать с этой информацией.

На самом деле я не использовал это на практике, а пример этого паттерна взял из библиотеки [дифференциальной эволюции](https://github.com/optozorax/differential-evolution/blob/9cfefc0b36ed3bf4ca6c37d786244d56c0b3b2ba/de_test/tutorial.cpp#L71).

Там же можно увидеть **паттерн termination strategy**, при помощи которого можно передавать в метод функцию, которая будет решать когда завершать метод. Но в рамках нашей учебной программы это излишная абстракция, и различные стратегии завершения метода никогда не применятся.

# Инкапсулируем с помощью `std::function`

В УМФ у нас есть задача: получить правую часть дифференциального уравнения, сетку конечных элементов, и на основе этой информации найти конечно-элементную аппроксимацию решения. Учитывая эту информацию, решатель МКЭ может выглядеть следующим образом:

```c++
vector<vector_t> solve_differential_equation(
	const function_3d_t& f,
	const grid_t& grid
);
```

Но при решении МКЭ мы не можем игнорировать ту вещь, что нам нужно выставлять краевые условия. Краевые условия - это известные нам значения функции на краях области, не имея краевых значений, невозможно решить задачу, она просто не сойдется.

Поэтому мы должны каким-то образом внутри функции решения дифференциального уравнения выставлять краевые условия.

В нашем курсе УМФ мы не решаем реальные задачи, а лишь исследуем насколько хорошо метод справляется с известными нам функциями `u`, поэтому может появиться соблазн написать следующим образом:

```c++
vector<vector_t> solve_differential_equation(
	const function_3d_t& f,
	const grid_t& grid,
	const function_3d_t& true_function_u
) {
	// doing smth
	// выставляем краевые условия с помощью функции u по краям области
	// делаем что-то дальше
}
```

Но это ужасный стиль! Зачем нам находить значение функции при помощи численных методов, которую мы уже знаем? Это выглядит как бред, поэтому более красивым может быть передавать **функцию, которая выставляет краевые условия**:

```c++
vector<vector_t> solve_differential_equation(
	const function_3d_t& f,
	const grid_t& grid, 
	const boundary_setter_function_t& set_boundary_conditions
) {
	// doing smth
	set_boundary_conditions(/* ... */);
	// делаем что-то дальше
}
```

Это очень красивое решение, которое может быть применено при решении реальных задач, если мы реально не знаем истинную функцию, но знаем значения краевых условий. Это значительно повышает абстрактность кода, позволяет его использовать в других проектах.

Так как вы можете использовать это на практике, более подробно смотрите код из курсовой по УМФ:
* [Прототип функции выставления краевых условий.](https://github.com/optozorax/labs_emf/blob/60b62fb5746aa83e3c52a802d5da1741950b6292/coursework/fem.h#L146)
* [Сама функция выставления краевых условий на основе известной нам функции u.](https://github.com/optozorax/labs_emf/blob/60b62fb5746aa83e3c52a802d5da1741950b6292/coursework/fem.h#L149)
* [Функция решения МКЭ.](https://github.com/optozorax/labs_emf/blob/60b62fb5746aa83e3c52a802d5da1741950b6292/coursework/fem.h#L159)
* [Пример использования.](https://github.com/optozorax/labs_emf/blob/60b62fb5746aa83e3c52a802d5da1741950b6292/coursework/main.cpp#L32)

# Заключение

Было показано каким образом `std::function` позволит вам избавиться от лишнего копипаста и инкапсулировать код и данные и в принципе сделать код намного более абстрактным и красивым.