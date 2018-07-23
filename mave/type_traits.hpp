#ifndef MAVE_TYPE_TRAITS_HPP
#define MAVE_TYPE_TRAITS_HPP
#include <type_traits>

namespace mave
{

template<typename T, std::size_t R, std::size_t C>
struct matrix;

template<typename T>
struct is_matrix : std::false_type {};
template<typename T, std::size_t R, std::size_t C>
struct is_matrix<matrix<T, R, C>> : std::true_type {};

template<typename T>
struct is_vector : std::false_type {};
template<typename T, std::size_t N>
struct is_vector<matrix<T, N, 1>> : std::true_type {};

// c++17 features ------------------------------------------------------------

template<typename ...>
struct conjunction : std::true_type{};
template<typename T>
struct conjunction<T> : T
{
    static_assert(std::is_convertible<decltype(T::value), bool>::value,
                  "conjunction<T> requires T::value is convertible to bool");
};
template<typename T, typename ... Ts>
struct conjunction<T, Ts...> :
    std::conditional<static_cast<bool>(T::value), conjunction<Ts...>, T>::type
{};

template<typename ...>
struct disjunction : std::false_type{};
template<typename T>
struct disjunction<T> : T
{
    static_assert(std::is_convertible<decltype(T::value), bool>::value,
                  "disjunction<T> requires T::value is convertible to bool");
};
template<typename T, typename ... Ts>
struct disjunction<T, Ts...> :
    std::conditional<static_cast<bool>(T::value), T, disjunction<Ts...>>::type
{};

template<typename T>
struct negation : std::integral_constant<bool, !static_cast<bool>(T::value)>{};

} // mave
#endif // MAVE_TYPE_TRAITS_HPP
