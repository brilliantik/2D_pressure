#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <string>
#include <complex>
#include <vector>

using namespace std;
using namespace std::complex_literals;

//I сценарий в эллипсе приближенными формулами для R>3H
//II сценарий в прямоугольнике приближенными формулами
//III сценарий в эллипсе аналитиечскими формулами (функция для ширин и длин выведена аналитически)

//S1 левый берег
//S2 правый берег

//------------Определение функций
double L1(double x, double R, double H, double L);
double L2(double x, double R, double H, double L);
double j1(double x, double R, double H, double L);
double j2(double x, double R, double H, double L);
double pod_int_func(double x, double P, double R);
complex<double> pod_int_func_elliptic(double x, complex<double> theta, double R);

double pod_int_func_2(double x, double R, double H, double L, double l);
double simpson(double x, double R);
double simpson_2(double x, double R, double H, double L);
complex<double> simpson_elliptic(double x, double R, double H, double L);

double a(double x, double R, double L);
double b(double x, double R, double L);
double c(double x, double R, double L);
double dd(double x, double R, double L);
double s(double x, double R, double L);
double A(double x, double R, double L);

//------------Определение указателей на функции
//double (*j_minus)(double x, double R, double H, double L);
//double (*j_plus)(double x, double R, double H, double L);
//double (*Lambda_minus)(double x, double R, double H, double L);
//double (*Lambda_plus)(double x, double R, double H, double L);

double pi = 2 * asin(1);//------------ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ определение числа пи

/// <summary>
/// Функция для вычисления длины трубок тока I сценария через интеграл 1/w(l)
/// </summary>
/// <param name="x --">Текущая координата х вдоль трещины</param>
/// <param name="R --">Характерный радиус контура питания для данного сценария</param>
/// <param name="H --">Длина трещины H=1 (!!! всегда)</param>
/// <param name="L --">Характерный полушаг между трещинами для данного сценария (!!! в данном сценарии не используется)</param>
/// <returns>Длина трубки тока при заданных параметрах для I сценария</returns>
double L1(double x, double R, double H, double L)
{
    complex<double> lambda;
    double ro = R + sqrt(R * R - 1);
    lambda = -(1.0i) * simpson_elliptic(x, R, H, L) * sqrt((1.0 - x * x) * (cosh(2.0 * log(ro)) - cos(2.0 * acos(x))) / (2.0 * (1.0 - x * x) + 2.0 * sinh(log(ro)) * sinh(log(ro))));
    return real(lambda);
}

/// <summary>
/// Функция для вычисления длины трубок тока II сценария через интеграл 1/w(l)
/// </summary>
/// <param name="x --">Текущая координата х вдоль трещины</param>
/// <param name="R --">Характерный радиус контура питания для данного сценария (!!! в данном сценарии не используется, а вычисляется)</param>
/// <param name="H --">Длина трещины H=1 (!!! всегда)</param>
/// <param name="L --">Характерный полушаг между трещинами для данного сценария</param>
/// <returns>Длина трубки тока при заданных параметрах для II сценария</returns>
double L2(double x, double R, double H, double L)
{
    double RR = 1.0 + 0.75 * L;
    return (1.0 - x) * (1.0 + 0.9 * L) + (RR - 1.0);
    //return (1.0 - x) * (1 - 0.05 * RR + 0.95 * L) + 1.005 * (RR - 1.0);
}

/// <summary>
/// Функция для вычисления притока I cценария через интеграл 1/w(l)
/// </summary>
/// <param name="x --">Текущая координата х вдоль трещины</param>
/// <param name="R --">Характерный радиус контура питания для данного сценария</param>
/// <param name="H --">Длина трещины H=1 (!!! всегда)</param>
/// <param name="L --">Характерный полушаг между трещинами для данного сценария (!!! в данном сценарии не используется)</param>
/// <returns>Приток для I сценария</returns>
double j1(double x, double R, double H, double L)
{
    double j;
    j = (1.0) * simpson(x, R) / (L1(x, R, H, L));
    return j;
}

/// <summary>
///Функция для вычисления притока II cценария через интеграл 1/w(l)
/// </summary>
/// <param name="x --">Текущая координата х вдоль трещины</param>
/// <param name="R --">Характерный радиус контура питания для данного сценария (!!! в данном сценарии не используется, а вычисляется)</param>
/// <param name="H --">Длина трещины H=1 (!!! всегда)</param>
/// <param name="L --">Характерный полушаг между трещинами для данного сценария</param>
/// <returns>Приток для II сценария</returns>
double j2(double x, double R, double H, double L)
{
    double RR = 1.0 + 0.75 * L;
    double j = L2(x, RR, H, L) * simpson_2(x, RR, H, L);
    return 1.0/j;
}

/// <summary>
/// Подинтегральное выражение в комплексных переменных для I сценария
/// </summary>
/// <param name="x --">Текущая координата х вдоль трещины</param>
/// <param name="P --">Текущее давление при заданной координате х</param>
/// <param name="R --">Характерный радиус контура питания</param>
/// <returns>Значение подинтегральной функции при заданных параметрах для I сценария</returns>
double pod_int_func(double x, double P, double R)
{
    ////double x = (x_1);

    //std::cout << std::fixed << std::setprecision(10);

    //std::complex<double> cheslitel_prav = sqrt((cosh(2.0 * P * log(R + sqrt(R * R - 1.0))) - cos((2.0 * acos(x) * log(R + sqrt(R * R - 1.0))) / (acosh(R)))) * log(R + sqrt(R * R - 1.0)) * log(R + sqrt(R * R - 1.0)));
    ////std::cout << "cheslitel_prav=" << cheslitel_prav << '\n';

    //std::complex<double> cheslitel_lev = abs((1.0) / (log(R + sqrt(R * R - 1.0)) * sqrt(-1.0 + (cos((acos(x) * log(R + sqrt(R * R - 1))) / (acosh(R))) * cosh(P * log(R + sqrt(R * R - 1))) + 1i * sin((acos(x) * log(R + sqrt(R * R - 1))) / (acosh(R))) * sinh(P * log(R + sqrt(R * R - 1)))) * (cos((acos(x) * log(R + sqrt(R * R - 1))) / (acosh(R))) * cosh(P * log(R + sqrt(R * R - 1))) + 1i * sin((acos(x) * log(R + sqrt(R * R - 1))) / (acosh(R))) * sinh(P * log(R + sqrt(R * R - 1)))))));
    ////std::cout << "cheslitel_lev=" << cheslitel_lev << '\n';

    //std::complex<double> z1 = -1.0 + (cos((acos(x) * log(R + sqrt(R * R - 1.0))) / (acosh(R)))) * (cos((acos(x) * log(R + sqrt(R * R - 1.0))) / (acosh(R))));
    ////std::cout << "z1=" << z1 << '\n';

    //std::complex<double> znamenatel = sqrt(2.0) * abs((1.0) / (sqrt(z1) * log(R + sqrt(R * R - 1.0))));
    ////std::cout << "znamenatel=" << znamenatel << '\n';

    ////std::cout << "result=" << real((cheslitel_lev * cheslitel_prav) / (znamenatel));
    //return real((cheslitel_lev * cheslitel_prav) / (znamenatel));

    return abs(sin(acos(x) + 1.0i * P * log(R + sqrt(R * R - 1)))) / sqrt(1 - x * x);
}

/// <summary>
/// Инегрирование вдоль трубки тока для I сценария (p_w=-1, p_Г=0, pod_int_func, квадратурная формула Симпсона)
/// </summary>
/// <param name="x --">Текущая координата х вдоль трещины</param>
/// <param name="R --">Характерный радиус контура питания</param>
/// <returns>Значение интгерала при заданных параметрах для I сценария</returns>
double simpson(double x, double R)
{
    double a = 0.0;
    double b = 1.0;
    double value;
    double sum = 0.0;

    double hh = 0.1;
    int N = (1 / hh) + 1;

    vector<double> M(N);
    for (int i = 0; i < N; i++)
    {
        M[i] = a + i * hh;
    }
    for (int i = 1; i < N; i++)
    {
        sum += (b - a) / (6.0 * (N - 1.0)) * (pod_int_func(x, M[i - 1], R) + 4.0 * pod_int_func(x, (M[i] + M[i - 1]) / (2.0), R) + pod_int_func(x, M[i], R));;
    }
    value = sum;

    return value;
    M.clear();
}

/// <summary>
/// Подинтегральное выражение для эллиптического интеграла 2-го рода !!!(как в вольфраме)
/// </summary>
/// <param name="x --">Текущая координата х вдоль трещины</param>
/// <param name="theta --">Переменная подинтегрального выражения</param>
/// <param name="R --">Характерный радиус контура питания</param>
/// <returns>Значение подинтегральной функции при заданных параметрах для I сценария</returns>
complex<double> pod_int_func_elliptic(double x, complex<double> theta, double R)
{
    return sqrt(1.0 - (1.0 / (1.0 - x * x)) * sin(theta) * sin(theta));
}

/// <summary>
/// Эллиптический интеграл 2-го рода !!!(как в вольфраме)
/// </summary>
/// <param name="x --">Текущая координата х вдоль трещины</param>
/// <param name="R --">Характерный радиус контура питания для данного сценария (!!! в данном сценарии не используется, а вычисляется)</param>
/// <param name="H --">Длина трещины H=1 (!!! всегда)</param>
/// <param name="L --">Характерный полушаг между трещинами для данного сценария</param>
/// <returns>Значение эллиптического интеграла при заданных параметрах</returns>
complex<double> simpson_elliptic(double x, double R, double H, double L)
{
    complex<double> a = 0.0;
    complex<double> b = 1.0i * log(R + sqrt(R * R - 1));
    complex<double> value;
    complex<double> sum = 0.0;

    int N = 100 + 1;
    complex<double> hh = (b - a) / (N - 1.0);

    vector<complex<double>> M(N);
    for (int j = 0; j < N; j++)
    {
        M[j] = a + j * 1.0 * hh;
    }
    for (int j = 1; j < N; j++)
    {
        sum += (b - a) / (6.0 * (N - 1.0)) * (pod_int_func_elliptic(x, M[j - 1], R) + 4.0 * pod_int_func_elliptic(x, (M[j] + M[j - 1]) / (2.0), R) + pod_int_func_elliptic(x, M[j], R));
    }
    value = sum;

    return value;
    M.clear();
}

/// <summary>
/// Подинтегральное выражение в комплексных переменных для II сценария
/// </summary>
/// <param name="x --">Текущая координата х вдоль трещины</param>
/// <param name="R --">Характерный радиус контура питания для данного сценария (!!! в данном сценарии не используется, а вычисляется)</param>
/// <param name="H --">Длина трещины H=1 (!!! всегда)</param>
/// <param name="L --">Характерный полушаг между трещинами для данного сценария</param>
/// <param name="l --">Текущая координата l вдоль трубки тока</param>
/// <returns>Значение подинтегральной функции при заданных параметрах для II сценария</returns>
double pod_int_func_2(double x, double R, double H, double L, double l)
{
    double w = (1.0) / (2.0) * (1.0 + c(x, R, L) + (b(x, R, L) * sqrt(2)) / (s(x, R, L) * l * dd(x, R, L) * sqrt(pi)) * exp(-(s(x, R, L) * s(x, R, L) - log(l * dd(x, R, L) / a(x, R, L))) * (s(x, R, L) * s(x, R, L) - log(l * dd(x, R, L) / a(x, R, L))) / (2.0 * s(x, R, L) * s(x, R, L))) + (c(x, R, L) - 1.0) * tanh(A(x, R, L) * log(l * dd(x, R, L) / a(x, R, L))));
    return 1.0 / w;
}

/// <summary>
/// Инегрирование вдоль трубки тока для II сценария (p_w=-1, p_Г=0, pod_int_func, квадратурная формула Симпсона)
/// </summary>
/// <param name="x --">Текущая координата х вдоль трещины</param>
/// <param name="R --">Характерный радиус контура питания для данного сценария (!!! в данном сценарии не используется, а вычисляется)</param>
/// <param name="H --">Длина трещины H=1 (!!! всегда)</param>
/// <param name="L --">Характерный полушаг между трещинами для данного сценария</param>
/// <returns>Значение интгерала при заданных параметрах для II сценария</returns>
double simpson_2(double x, double R, double H, double L)
{
    double a = 0.000000000000001; //особенность в 0
    double b = 1.0;
    double value;
    double sum = 0.0;

    int N = 100 + 1;
    double hh = (b - a) / (N - 1.0);

    double* M = new double[N];
    for (int i = 0; i < N; i++)
    {
        M[i] = a + i * hh;
    }
    for (int i = 1; i < N; i++)
    {
        sum += (b - a) / (6.0 * (N - 1.0)) * (pod_int_func_2(x, R, H, L, M[i - 1]) + 4.0 * pod_int_func_2(x, R, H, L, (M[i] + M[i - 1]) / (2.0)) + pod_int_func_2(x, R, H, L, M[i]));;
    }
    value = sum;

    return value;
    delete[] M;
}

/// <summary>
/// Вычисление вспомогательного параметра a для II сценария
/// </summary>
/// <param name="x --">Текущая координата х вдоль трещины</param>
/// <param name="R --">Характерный радиус контура питания для данного сценария (!!! в данном сценарии не используется, а вычисляется)</param>
/// <param name="L --">Характерный полушаг между трещинами для данного сценария</param>
/// <returns>Значение вспомогательного параметра a для II сценария</returns>
double a(double x, double R, double L)
{
    return ((11.0) * (1.0 - x) * log(22.0 * L)) / (95.0) + (9.0 * x) / (19.0);
}

/// <summary>
/// Вычисление вспомогательного параметра b для II сценария
/// </summary>
/// <param name="x --">Текущая координата х вдоль трещины</param>
/// <param name="R --">Характерный радиус контура питания для данного сценария (!!! в данном сценарии не используется, а вычисляется)</param>
/// <param name="L --">Характерный полушаг между трещинами для данного сценария</param>
/// <returns>Значение вспомогательного параметра b для II сценария</returns>
double b(double x, double R, double L)
{
    return (1.0) / (10.0) * (1.0 / sqrt(1 - x) + 1.0 / (0.1 + x)) + (0.29 * L - 0.56);
}

double dd(double x, double R, double L)
{
    return 1.0 + 2.985 * (L - 0.5) * pow(x, 15);
}

/// <summary>
/// Вычисление вспомогательного параметра c для II сценария
/// </summary>
/// <param name="x --">Текущая координата х вдоль трещины</param>
/// <param name="R --">Характерный радиус контура питания для данного сценария (!!! в данном сценарии не используется, а вычисляется)</param>
/// <param name="L --">Характерный полушаг между трещинами для данного сценария</param>
/// <returns>Значение вспомогательного параметра c для II сценария</returns>
double c(double x, double R, double L)
{
    return (1.0) / (15.0 * (1 - pow(x, 1.5))) + (0.72 * L - 0.2);
}

/// <summary>
/// Вычисление вспомогательного параметра s для II сценария
/// </summary>
/// <param name="x --">Текущая координата х вдоль трещины</param>
/// <param name="R --">Характерный радиус контура питания для данного сценария (!!! в данном сценарии не используется, а вычисляется)</param>
/// <param name="L --">Характерный полушаг между трещинами для данного сценария</param>
/// <returns>Значение вспомогательного параметра s для II сценария</returns>
double s(double x, double R, double L)
{
    return (1.0) * (4.0 * L * (x - 1.0) + 9.6 * (x + 1.0)) / (19.0);
}

/// <summary>
/// Вычисление вспомогательного параметра A для II сценария
/// </summary>
/// <param name="x --">Текущая координата х вдоль трещины</param>
/// <param name="R --">Характерный радиус контура питания для данного сценария (!!! в данном сценарии не используется, а вычисляется)</param>
/// <param name="L --">Характерный полушаг между трещинами для данного сценария</param>
/// <returns>Значение вспомогательного параметра A для II сценария</returns>
double A(double x, double R, double L)
{
    double alpha = 0.0;
    if (L <= 0.5)
    {
        alpha = 0.8 * L + 0.025;
    }
    else if (L > 0.5)
    {
        alpha = 0.45;
    }
    else
    {
        cout << "Error param alpha";
    }
    return pow(x, -alpha);
}