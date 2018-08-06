#ifndef MAVE_FORWARD_HPP
#define MAVE_FORWARD_HPP
#include "matrix.hpp"
#include "vector.hpp"

namespace mave
{

#undef  MAVE_GENERATE_FORWARDING_UNARY_FUNCTIONS_MATRIX
#define MAVE_GENERATE_FORWARDING_UNARY_FUNCTIONS_MATRIX(FUNC_NAME, MODIFICATION)\
    template<typename T, std::size_t R, std::size_t C>\
    MAVE_INLINE std::pair<matrix<T, R, C>, matrix<T, R, C>>\
    FUNC_NAME(std::tuple<matrix<T, R, C> MODIFICATION,\
                         matrix<T, R, C> MODIFICATION> ms) noexcept\
    {\
        return FUNC_NAME(\
            std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&>(\
                std::get<0>(ms), std::get<1>(ms)));\
    }\
    template<typename T, std::size_t R, std::size_t C>\
    MAVE_INLINE std::tuple<matrix<T, R, C>, matrix<T, R, C>, matrix<T, R, C>>\
    FUNC_NAME(std::tuple<matrix<T, R, C> MODIFICATION,\
                         matrix<T, R, C> MODIFICATION,\
                         matrix<T, R, C> MODIFICATION \
                         > ms) noexcept\
    {\
        return FUNC_NAME(\
            std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&,\
                       const matrix<T, R, C>&>(\
                std::get<0>(ms), std::get<1>(ms), std::get<2>(ms)));\
    }\
    template<typename T, std::size_t R, std::size_t C>\
    MAVE_INLINE std::tuple<matrix<T, R, C>, matrix<T, R, C>,\
                           matrix<T, R, C>, matrix<T, R, C>>\
    FUNC_NAME(std::tuple<matrix<T, R, C> MODIFICATION,\
                         matrix<T, R, C> MODIFICATION,\
                         matrix<T, R, C> MODIFICATION,\
                         matrix<T, R, C> MODIFICATION \
                         > ms) noexcept\
    {\
        return FUNC_NAME(\
            std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&,\
                       const matrix<T, R, C>&, const matrix<T, R, C>&>(\
            std::get<0>(ms), std::get<1>(ms), std::get<2>(ms), std::get<3>(ms)));\
    }\
    /**/

MAVE_GENERATE_FORWARDING_UNARY_FUNCTIONS_MATRIX(operator-, MAVE_EMPTY_ARGUMENT)
MAVE_GENERATE_FORWARDING_UNARY_FUNCTIONS_MATRIX(operator-, &)
#undef MAVE_GENERATE_FORWARDING_UNARY_FUNCTIONS_MATRIX


#undef  MAVE_GENERATE_EXPANDING_UNARY_FUNCTIONS_MATRIX
#define MAVE_GENERATE_EXPANDING_UNARY_FUNCTIONS_MATRIX(FUNC_NAME, MODIFICATION)\
    template<typename T, std::size_t R, std::size_t C>\
    MAVE_INLINE std::pair<matrix<T, R, C>, matrix<T, R, C>>\
    FUNC_NAME(std::tuple<matrix<T, R, C> MODIFICATION,\
                         matrix<T, R, C> MODIFICATION> ms) noexcept\
    {\
        return FUNC_NAME(std::get<0>(ms), std::get<1>(ms));\
    }\
    template<typename T, std::size_t R, std::size_t C>\
    MAVE_INLINE std::pair<matrix<T, R, C>, matrix<T, R, C>>\
    FUNC_NAME(std::pair<matrix<T, R, C> MODIFICATION,\
                        matrix<T, R, C> MODIFICATION> ms) noexcept\
    {\
        return FUNC_NAME(std::get<0>(ms), std::get<1>(ms));\
    }\
    template<typename T, std::size_t R, std::size_t C>\
    MAVE_INLINE std::tuple<matrix<T, R, C>, matrix<T, R, C>, matrix<T, R, C>>\
    FUNC_NAME(std::tuple<matrix<T, R, C> MODIFICATION,\
                         matrix<T, R, C> MODIFICATION,\
                         matrix<T, R, C> MODIFICATION \
                         > ms) noexcept\
    {\
        return FUNC_NAME(std::get<0>(ms), std::get<1>(ms), std::get<2>(ms));\
    }\
    template<typename T, std::size_t R, std::size_t C>\
    MAVE_INLINE std::tuple<matrix<T, R, C>, matrix<T, R, C>,\
                           matrix<T, R, C>, matrix<T, R, C>>\
    FUNC_NAME(std::tuple<matrix<T, R, C> MODIFICATION,\
                         matrix<T, R, C> MODIFICATION,\
                         matrix<T, R, C> MODIFICATION,\
                         matrix<T, R, C> MODIFICATION \
                         > ms) noexcept\
    {\
        return FUNC_NAME(std::get<0>(ms), std::get<1>(ms),\
                         std::get<2>(ms), std::get<3>(ms));\
    }\
    /**/

MAVE_GENERATE_EXPANDING_UNARY_FUNCTIONS_MATRIX(floor,  MAVE_EMPTY_ARGUMENT)
MAVE_GENERATE_EXPANDING_UNARY_FUNCTIONS_MATRIX(floor,  const&)
MAVE_GENERATE_EXPANDING_UNARY_FUNCTIONS_MATRIX(floor,  &)
MAVE_GENERATE_EXPANDING_UNARY_FUNCTIONS_MATRIX(ceil,   MAVE_EMPTY_ARGUMENT)
MAVE_GENERATE_EXPANDING_UNARY_FUNCTIONS_MATRIX(ceil,   const&)
MAVE_GENERATE_EXPANDING_UNARY_FUNCTIONS_MATRIX(ceil,   &)
#undef MAVE_GENERATE_EXPANDING_UNARY_FUNCTIONS_MATRIX



#undef  MAVE_GENERATE_FORWARDING_OP_ASSIGN_MATRIX_MATRIX
#define MAVE_GENERATE_FORWARDING_OP_ASSIGN_MATRIX_MATRIX(FUNC_NAME, MODIFICATION)\
    template<typename T, std::size_t R, std::size_t C>\
    MAVE_INLINE void\
    FUNC_NAME(std::tuple<matrix<T, R, C>&, matrix<T, R, C>&> lhs,\
              std::tuple<matrix<T, R, C> MODIFICATION,\
                         matrix<T, R, C> MODIFICATION> rhs) noexcept\
    {\
        FUNC_NAME(lhs,\
            std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&>(\
                std::get<0>(rhs), std::get<1>(rhs)));\
        return;\
    }\
    template<typename T, std::size_t R, std::size_t C>\
    MAVE_INLINE void\
    FUNC_NAME(std::tuple<matrix<T, R, C>&, matrix<T, R, C>&,\
                         matrix<T, R, C>&> lhs,\
              std::tuple<matrix<T, R, C> MODIFICATION,\
                         matrix<T, R, C> MODIFICATION,\
                         matrix<T, R, C> MODIFICATION \
                         > rhs) noexcept\
    {\
        FUNC_NAME(lhs,\
            std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&,\
                       const matrix<T, R, C>&>(\
                std::get<0>(rhs), std::get<1>(rhs), std::get<2>(rhs)));\
        return;\
    }\
    template<typename T, std::size_t R, std::size_t C>\
    MAVE_INLINE void\
    FUNC_NAME(std::tuple<matrix<T, R, C>&, matrix<T, R, C>&,\
                         matrix<T, R, C>&, matrix<T, R, C>&> lhs,\
              std::tuple<matrix<T, R, C> MODIFICATION,\
                         matrix<T, R, C> MODIFICATION,\
                         matrix<T, R, C> MODIFICATION,\
                         matrix<T, R, C> MODIFICATION \
                         > rhs) noexcept\
    {\
        FUNC_NAME(lhs,\
            std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&,\
                       const matrix<T, R, C>&, const matrix<T, R, C>&>(\
            std::get<0>(rhs), std::get<1>(rhs), std::get<2>(rhs), std::get<3>(rhs)));\
        return;\
    }\
    /**/

MAVE_GENERATE_FORWARDING_OP_ASSIGN_MATRIX_MATRIX(operator+=, MAVE_EMPTY_ARGUMENT)
MAVE_GENERATE_FORWARDING_OP_ASSIGN_MATRIX_MATRIX(operator+=, &)
MAVE_GENERATE_FORWARDING_OP_ASSIGN_MATRIX_MATRIX(operator-=, MAVE_EMPTY_ARGUMENT)
MAVE_GENERATE_FORWARDING_OP_ASSIGN_MATRIX_MATRIX(operator-=, &)
#undef MAVE_GENERATE_FORWARDING_OP_ASSIGN_MATRIX_MATRIX



#undef  MAVE_GENERATE_FORWARDING_OP_ASSIGN_MATRIX_SCALAR
#define MAVE_GENERATE_FORWARDING_OP_ASSIGN_MATRIX_SCALAR(FUNC_NAME, MODIFICATION)\
    template<typename T, std::size_t R, std::size_t C>\
    MAVE_INLINE void\
    FUNC_NAME(std::tuple<matrix<T, R, C>&, matrix<T, R, C>&> lhs,\
              std::tuple<T MODIFICATION,\
                         T MODIFICATION> rhs) noexcept\
    {\
        FUNC_NAME(lhs, std::tuple<T, T>(std::get<0>(rhs), std::get<1>(rhs)));\
        return;\
    }\
    template<typename T, std::size_t R, std::size_t C>\
    MAVE_INLINE void\
    FUNC_NAME(std::tuple<matrix<T, R, C>&, matrix<T, R, C>&,\
                         matrix<T, R, C>&> lhs,\
              std::tuple<T MODIFICATION,\
                         T MODIFICATION,\
                         T MODIFICATION \
                         > rhs) noexcept\
    {\
        FUNC_NAME(lhs, std::tuple<T, T, T>(\
                    std::get<0>(rhs), std::get<1>(rhs), std::get<2>(rhs)));\
        return;\
    }\
    template<typename T, std::size_t R, std::size_t C>\
    MAVE_INLINE void\
    FUNC_NAME(std::tuple<matrix<T, R, C>&, matrix<T, R, C>&,\
                         matrix<T, R, C>&, matrix<T, R, C>&> lhs,\
              std::tuple<T MODIFICATION,\
                         T MODIFICATION,\
                         T MODIFICATION,\
                         T MODIFICATION \
                         > rhs) noexcept\
    {\
        FUNC_NAME(lhs, std::tuple<T, T, T, T>(\
                    std::get<0>(rhs), std::get<1>(rhs),\
                    std::get<2>(rhs), std::get<3>(rhs)));\
        return;\
    }\
    /**/

MAVE_GENERATE_FORWARDING_OP_ASSIGN_MATRIX_SCALAR(operator*=, const&)
MAVE_GENERATE_FORWARDING_OP_ASSIGN_MATRIX_SCALAR(operator*=, &)
MAVE_GENERATE_FORWARDING_OP_ASSIGN_MATRIX_SCALAR(operator/=, const&)
MAVE_GENERATE_FORWARDING_OP_ASSIGN_MATRIX_SCALAR(operator/=, &)
#undef MAVE_GENERATE_FORWARDING_OP_ASSIGN_MATRIX_SCALAR



#undef  MAVE_GENERATE_FORWARDING_BINARY_FUNCTIONS_MATRIX_MATRIX_MATRIX
#define MAVE_GENERATE_FORWARDING_BINARY_FUNCTIONS_MATRIX_MATRIX_MATRIX(FUNC_NAME, L_MODIFICATION, R_MODIFICATION)\
    template<typename T, std::size_t R, std::size_t C>\
    MAVE_INLINE std::pair<matrix<T, R, C>, matrix<T, R, C>>\
    FUNC_NAME(std::tuple<matrix<T, R, C> L_MODIFICATION,\
                         matrix<T, R, C> L_MODIFICATION \
                         > lhs,\
              std::tuple<matrix<T, R, C> R_MODIFICATION,\
                         matrix<T, R, C> R_MODIFICATION \
                         > rhs) noexcept\
    {\
        return FUNC_NAME(\
            std::tuple<matrix<T, R, C> const&, matrix<T, R, C> const&>(\
                std::get<0>(lhs), std::get<1>(lhs)),                         \
            std::tuple<matrix<T, R, C> const&, matrix<T, R, C> const&>(\
                std::get<0>(rhs), std::get<1>(rhs)));                        \
    }\
    template<typename T, std::size_t R, std::size_t C>\
    MAVE_INLINE std::tuple<matrix<T, R, C>, matrix<T, R, C>, matrix<T, R, C>>\
    FUNC_NAME(std::tuple<matrix<T, R, C> L_MODIFICATION,\
                         matrix<T, R, C> L_MODIFICATION,\
                         matrix<T, R, C> L_MODIFICATION \
                         > lhs,\
              std::tuple<matrix<T, R, C> R_MODIFICATION,\
                         matrix<T, R, C> R_MODIFICATION,\
                         matrix<T, R, C> R_MODIFICATION \
                         > rhs) noexcept\
    {\
        return FUNC_NAME(\
            std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&,\
                       const matrix<T, R, C>&>(                       \
                std::get<0>(lhs), std::get<1>(lhs), std::get<2>(lhs)),   \
            std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&,\
                       const matrix<T, R, C>&>(                       \
                std::get<0>(rhs), std::get<1>(rhs), std::get<2>(rhs)));  \
    }\
    template<typename T, std::size_t R, std::size_t C>\
    MAVE_INLINE std::tuple<matrix<T, R, C>, matrix<T, R, C>,\
                           matrix<T, R, C>, matrix<T, R, C>>\
    FUNC_NAME(std::tuple<matrix<T, R, C> L_MODIFICATION,\
                         matrix<T, R, C> L_MODIFICATION,\
                         matrix<T, R, C> L_MODIFICATION,\
                         matrix<T, R, C> L_MODIFICATION \
                         > lhs,\
              std::tuple<matrix<T, R, C> R_MODIFICATION,\
                         matrix<T, R, C> R_MODIFICATION,\
                         matrix<T, R, C> R_MODIFICATION,\
                         matrix<T, R, C> R_MODIFICATION \
                         > rhs) noexcept\
    {\
        return FUNC_NAME(\
            std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&, \
                       const matrix<T, R, C>&, const matrix<T, R, C>&>(\
            std::get<0>(lhs), std::get<1>(lhs), std::get<2>(lhs), std::get<3>(lhs)),\
            std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&, \
                       const matrix<T, R, C>&, const matrix<T, R, C>&>(\
            std::get<0>(rhs), std::get<1>(rhs), std::get<2>(rhs), std::get<3>(rhs)));\
    }\
    /**/

MAVE_GENERATE_FORWARDING_BINARY_FUNCTIONS_MATRIX_MATRIX_MATRIX(operator+, MAVE_EMPTY_ARGUMENT, MAVE_EMPTY_ARGUMENT)
MAVE_GENERATE_FORWARDING_BINARY_FUNCTIONS_MATRIX_MATRIX_MATRIX(operator+, const&             , MAVE_EMPTY_ARGUMENT)
MAVE_GENERATE_FORWARDING_BINARY_FUNCTIONS_MATRIX_MATRIX_MATRIX(operator+, &                  , MAVE_EMPTY_ARGUMENT)
MAVE_GENERATE_FORWARDING_BINARY_FUNCTIONS_MATRIX_MATRIX_MATRIX(operator+, MAVE_EMPTY_ARGUMENT, &)
MAVE_GENERATE_FORWARDING_BINARY_FUNCTIONS_MATRIX_MATRIX_MATRIX(operator+, const&             , &)
MAVE_GENERATE_FORWARDING_BINARY_FUNCTIONS_MATRIX_MATRIX_MATRIX(operator+, &                  , &)
MAVE_GENERATE_FORWARDING_BINARY_FUNCTIONS_MATRIX_MATRIX_MATRIX(operator+, MAVE_EMPTY_ARGUMENT, const&)
MAVE_GENERATE_FORWARDING_BINARY_FUNCTIONS_MATRIX_MATRIX_MATRIX(operator+, &                  , const&)

MAVE_GENERATE_FORWARDING_BINARY_FUNCTIONS_MATRIX_MATRIX_MATRIX(operator-, MAVE_EMPTY_ARGUMENT, MAVE_EMPTY_ARGUMENT)
MAVE_GENERATE_FORWARDING_BINARY_FUNCTIONS_MATRIX_MATRIX_MATRIX(operator-, const&             , MAVE_EMPTY_ARGUMENT)
MAVE_GENERATE_FORWARDING_BINARY_FUNCTIONS_MATRIX_MATRIX_MATRIX(operator-, &                  , MAVE_EMPTY_ARGUMENT)
MAVE_GENERATE_FORWARDING_BINARY_FUNCTIONS_MATRIX_MATRIX_MATRIX(operator-, MAVE_EMPTY_ARGUMENT, &)
MAVE_GENERATE_FORWARDING_BINARY_FUNCTIONS_MATRIX_MATRIX_MATRIX(operator-, const&             , &)
MAVE_GENERATE_FORWARDING_BINARY_FUNCTIONS_MATRIX_MATRIX_MATRIX(operator-, &                  , &)
MAVE_GENERATE_FORWARDING_BINARY_FUNCTIONS_MATRIX_MATRIX_MATRIX(operator-, MAVE_EMPTY_ARGUMENT, const&)
MAVE_GENERATE_FORWARDING_BINARY_FUNCTIONS_MATRIX_MATRIX_MATRIX(operator-, &                  , const&)

MAVE_GENERATE_FORWARDING_BINARY_FUNCTIONS_MATRIX_MATRIX_MATRIX(min,       MAVE_EMPTY_ARGUMENT, MAVE_EMPTY_ARGUMENT)
MAVE_GENERATE_FORWARDING_BINARY_FUNCTIONS_MATRIX_MATRIX_MATRIX(min,       const&             , MAVE_EMPTY_ARGUMENT)
MAVE_GENERATE_FORWARDING_BINARY_FUNCTIONS_MATRIX_MATRIX_MATRIX(min,       &                  , MAVE_EMPTY_ARGUMENT)
MAVE_GENERATE_FORWARDING_BINARY_FUNCTIONS_MATRIX_MATRIX_MATRIX(min,       MAVE_EMPTY_ARGUMENT, &)
MAVE_GENERATE_FORWARDING_BINARY_FUNCTIONS_MATRIX_MATRIX_MATRIX(min,       const&             , &)
MAVE_GENERATE_FORWARDING_BINARY_FUNCTIONS_MATRIX_MATRIX_MATRIX(min,       &                  , &)
MAVE_GENERATE_FORWARDING_BINARY_FUNCTIONS_MATRIX_MATRIX_MATRIX(min,       MAVE_EMPTY_ARGUMENT, const&)
MAVE_GENERATE_FORWARDING_BINARY_FUNCTIONS_MATRIX_MATRIX_MATRIX(min,       &                  , const&)

MAVE_GENERATE_FORWARDING_BINARY_FUNCTIONS_MATRIX_MATRIX_MATRIX(max,       MAVE_EMPTY_ARGUMENT, MAVE_EMPTY_ARGUMENT)
MAVE_GENERATE_FORWARDING_BINARY_FUNCTIONS_MATRIX_MATRIX_MATRIX(max,       const&             , MAVE_EMPTY_ARGUMENT)
MAVE_GENERATE_FORWARDING_BINARY_FUNCTIONS_MATRIX_MATRIX_MATRIX(max,       &                  , MAVE_EMPTY_ARGUMENT)
MAVE_GENERATE_FORWARDING_BINARY_FUNCTIONS_MATRIX_MATRIX_MATRIX(max,       MAVE_EMPTY_ARGUMENT, &)
MAVE_GENERATE_FORWARDING_BINARY_FUNCTIONS_MATRIX_MATRIX_MATRIX(max,       const&             , &)
MAVE_GENERATE_FORWARDING_BINARY_FUNCTIONS_MATRIX_MATRIX_MATRIX(max,       &                  , &)
MAVE_GENERATE_FORWARDING_BINARY_FUNCTIONS_MATRIX_MATRIX_MATRIX(max,       MAVE_EMPTY_ARGUMENT, const&)
MAVE_GENERATE_FORWARDING_BINARY_FUNCTIONS_MATRIX_MATRIX_MATRIX(max,       &                  , const&)
#undef MAVE_GENERATE_FORWARDING_BINARY_FUNCTIONS_MATRIX_MATRIX_MATRIX


#undef  MAVE_GENERATE_FORWARDING_BINARY_FUNCTIONS_MATRIX_MATRIX_SCALAR
#define MAVE_GENERATE_FORWARDING_BINARY_FUNCTIONS_MATRIX_MATRIX_SCALAR(FUNC_NAME, L_MODIFICATION, R_MODIFICATION)\
    template<typename T, std::size_t R, std::size_t C>\
    MAVE_INLINE std::pair<matrix<T, R, C>, matrix<T, R, C>>\
    FUNC_NAME(std::tuple<matrix<T, R, C> L_MODIFICATION,\
                         matrix<T, R, C> L_MODIFICATION \
                         > lhs,\
              std::tuple<T R_MODIFICATION,\
                         T R_MODIFICATION \
                         > rhs) noexcept\
    {\
        return FUNC_NAME(\
            std::tuple<matrix<T, R, C> const&, matrix<T, R, C> const&>(\
                std::get<0>(lhs), std::get<1>(lhs)),                         \
            std::tuple<matrix<T, R, C> const&, matrix<T, R, C> const&>(\
                std::get<0>(rhs), std::get<1>(rhs)));                        \
    }\
    template<typename T, std::size_t R, std::size_t C>\
    MAVE_INLINE std::tuple<matrix<T, R, C>, matrix<T, R, C>, matrix<T, R, C>>\
    FUNC_NAME(std::tuple<matrix<T, R, C> L_MODIFICATION,\
                         matrix<T, R, C> L_MODIFICATION,\
                         matrix<T, R, C> L_MODIFICATION \
                         > lhs,\
              std::tuple<T R_MODIFICATION,\
                         T R_MODIFICATION,\
                         T R_MODIFICATION \
                         > rhs) noexcept\
    {\
        return FUNC_NAME(\
            std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&,\
                       const matrix<T, R, C>&>(                       \
                std::get<0>(lhs), std::get<1>(lhs), std::get<2>(lhs)),   \
            std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&,\
                       const matrix<T, R, C>&>(                       \
                std::get<0>(rhs), std::get<1>(rhs), std::get<2>(rhs)));  \
    }\
    template<typename T, std::size_t R, std::size_t C>\
    MAVE_INLINE std::tuple<matrix<T, R, C>, matrix<T, R, C>,\
                           matrix<T, R, C>, matrix<T, R, C>>\
    FUNC_NAME(std::tuple<matrix<T, R, C> L_MODIFICATION,\
                         matrix<T, R, C> L_MODIFICATION,\
                         matrix<T, R, C> L_MODIFICATION,\
                         matrix<T, R, C> L_MODIFICATION \
                         > lhs,\
              std::tuple<T R_MODIFICATION,\
                         T R_MODIFICATION,\
                         T R_MODIFICATION,\
                         T R_MODIFICATION \
                         > rhs) noexcept\
    {\
        return FUNC_NAME(\
            std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&, \
                       const matrix<T, R, C>&, const matrix<T, R, C>&>(\
            std::get<0>(lhs), std::get<1>(lhs), std::get<2>(lhs), std::get<3>(lhs)),\
            std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&, \
                       const matrix<T, R, C>&, const matrix<T, R, C>&>(\
            std::get<0>(rhs), std::get<1>(rhs), std::get<2>(rhs), std::get<3>(rhs)));\
    }\
    /**/

MAVE_GENERATE_FORWARDING_BINARY_FUNCTIONS_MATRIX_MATRIX_SCALAR(operator*, MAVE_EMPTY_ARGUMENT, MAVE_EMPTY_ARGUMENT)
MAVE_GENERATE_FORWARDING_BINARY_FUNCTIONS_MATRIX_MATRIX_SCALAR(operator*, &                  , MAVE_EMPTY_ARGUMENT)
MAVE_GENERATE_FORWARDING_BINARY_FUNCTIONS_MATRIX_MATRIX_SCALAR(operator*, MAVE_EMPTY_ARGUMENT, &)
MAVE_GENERATE_FORWARDING_BINARY_FUNCTIONS_MATRIX_MATRIX_SCALAR(operator*, const&             , &)
MAVE_GENERATE_FORWARDING_BINARY_FUNCTIONS_MATRIX_MATRIX_SCALAR(operator*, &                  , &)
MAVE_GENERATE_FORWARDING_BINARY_FUNCTIONS_MATRIX_MATRIX_SCALAR(operator*, MAVE_EMPTY_ARGUMENT, const&)
MAVE_GENERATE_FORWARDING_BINARY_FUNCTIONS_MATRIX_MATRIX_SCALAR(operator*, const&,              const&)
MAVE_GENERATE_FORWARDING_BINARY_FUNCTIONS_MATRIX_MATRIX_SCALAR(operator*, &                  , const&)

MAVE_GENERATE_FORWARDING_BINARY_FUNCTIONS_MATRIX_MATRIX_SCALAR(operator/, MAVE_EMPTY_ARGUMENT, MAVE_EMPTY_ARGUMENT)
MAVE_GENERATE_FORWARDING_BINARY_FUNCTIONS_MATRIX_MATRIX_SCALAR(operator/, &                  , MAVE_EMPTY_ARGUMENT)
MAVE_GENERATE_FORWARDING_BINARY_FUNCTIONS_MATRIX_MATRIX_SCALAR(operator/, MAVE_EMPTY_ARGUMENT, &)
MAVE_GENERATE_FORWARDING_BINARY_FUNCTIONS_MATRIX_MATRIX_SCALAR(operator/, const&             , &)
MAVE_GENERATE_FORWARDING_BINARY_FUNCTIONS_MATRIX_MATRIX_SCALAR(operator/, &                  , &)
MAVE_GENERATE_FORWARDING_BINARY_FUNCTIONS_MATRIX_MATRIX_SCALAR(operator/, MAVE_EMPTY_ARGUMENT, const&)
MAVE_GENERATE_FORWARDING_BINARY_FUNCTIONS_MATRIX_MATRIX_SCALAR(operator/, const&,              const&)
MAVE_GENERATE_FORWARDING_BINARY_FUNCTIONS_MATRIX_MATRIX_SCALAR(operator/, &                  , const&)
#undef MAVE_GENERATE_FORWARDING_BINARY_FUNCTIONS_MATRIX_MATRIX_SCALAR



#undef  MAVE_GENERATE_FORWARDING_BINARY_FUNCTIONS_MATRIX_SCALAR_MATRIX
#define MAVE_GENERATE_FORWARDING_BINARY_FUNCTIONS_MATRIX_SCALAR_MATRIX(FUNC_NAME, L_MODIFICATION, R_MODIFICATION)\
    template<typename T, std::size_t R, std::size_t C>\
    MAVE_INLINE std::pair<matrix<T, R, C>, matrix<T, R, C>>\
    FUNC_NAME(std::tuple<T L_MODIFICATION,\
                         T L_MODIFICATION \
                         > lhs,\
              std::tuple<matrix<T, R, C> R_MODIFICATION,\
                         matrix<T, R, C> R_MODIFICATION \
                         > rhs) noexcept\
    {\
        return FUNC_NAME(\
            std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&>(\
                std::get<0>(lhs), std::get<1>(lhs)),                         \
            std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&>(\
                std::get<0>(rhs), std::get<1>(rhs)));                        \
    }\
    template<typename T, std::size_t R, std::size_t C>\
    MAVE_INLINE std::tuple<matrix<T, R, C>, matrix<T, R, C>, matrix<T, R, C>>\
    FUNC_NAME(std::tuple<T L_MODIFICATION,\
                         T L_MODIFICATION,\
                         T L_MODIFICATION \
                         > lhs,\
              std::tuple<matrix<T, R, C> R_MODIFICATION,\
                         matrix<T, R, C> R_MODIFICATION,\
                         matrix<T, R, C> R_MODIFICATION \
                         > rhs) noexcept\
    {\
        return FUNC_NAME(\
            std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&,\
                       const matrix<T, R, C>&>(                       \
                std::get<0>(lhs), std::get<1>(lhs), std::get<2>(lhs)),   \
            std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&,\
                       const matrix<T, R, C>&>(                       \
                std::get<0>(rhs), std::get<1>(rhs), std::get<2>(rhs)));  \
    }\
    template<typename T, std::size_t R, std::size_t C>\
    MAVE_INLINE std::tuple<matrix<T, R, C>, matrix<T, R, C>,\
                           matrix<T, R, C>, matrix<T, R, C>>\
    FUNC_NAME(std::tuple<T L_MODIFICATION,\
                         T L_MODIFICATION,\
                         T L_MODIFICATION,\
                         T L_MODIFICATION \
                         > lhs,\
              std::tuple<matrix<T, R, C> R_MODIFICATION,\
                         matrix<T, R, C> R_MODIFICATION,\
                         matrix<T, R, C> R_MODIFICATION,\
                         matrix<T, R, C> R_MODIFICATION \
                         > rhs) noexcept\
    {\
        return FUNC_NAME(\
            std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&, \
                       const matrix<T, R, C>&, const matrix<T, R, C>&>(\
            std::get<0>(lhs), std::get<1>(lhs), std::get<2>(lhs), std::get<3>(lhs)),\
            std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&, \
                       const matrix<T, R, C>&, const matrix<T, R, C>&>(\
            std::get<0>(rhs), std::get<1>(rhs), std::get<2>(rhs), std::get<3>(rhs)));\
    }\
    /**/

MAVE_GENERATE_FORWARDING_BINARY_FUNCTIONS_MATRIX_SCALAR_MATRIX(operator*, MAVE_EMPTY_ARGUMENT, MAVE_EMPTY_ARGUMENT)
MAVE_GENERATE_FORWARDING_BINARY_FUNCTIONS_MATRIX_SCALAR_MATRIX(operator*, const&             , MAVE_EMPTY_ARGUMENT)
MAVE_GENERATE_FORWARDING_BINARY_FUNCTIONS_MATRIX_SCALAR_MATRIX(operator*, &                  , MAVE_EMPTY_ARGUMENT)
MAVE_GENERATE_FORWARDING_BINARY_FUNCTIONS_MATRIX_SCALAR_MATRIX(operator*, MAVE_EMPTY_ARGUMENT, &)
MAVE_GENERATE_FORWARDING_BINARY_FUNCTIONS_MATRIX_SCALAR_MATRIX(operator*, const&             , &)
MAVE_GENERATE_FORWARDING_BINARY_FUNCTIONS_MATRIX_SCALAR_MATRIX(operator*, &                  , &)
MAVE_GENERATE_FORWARDING_BINARY_FUNCTIONS_MATRIX_SCALAR_MATRIX(operator*, const&,              const&)
MAVE_GENERATE_FORWARDING_BINARY_FUNCTIONS_MATRIX_SCALAR_MATRIX(operator*, &                  , const&)
#undef MAVE_GENERATE_FORWARDING_BINARY_FUNCTIONS_MATRIX_SCALAR_MATRIX

#undef  MAVE_GENERATE_FORWARDING_TERNARY_FUNCTIONS_MATRIX_SCALAR_MATRIX_MATRIX
#define MAVE_GENERATE_FORWARDING_TERNARY_FUNCTIONS_MATRIX_SCALAR_MATRIX_MATRIX(\
        FUNC_NAME, L_MODIFICATION, M_MODIFICATION, R_MODIFICATION)\
    template<typename T, std::size_t R, std::size_t C>\
    MAVE_INLINE std::pair<matrix<T, R, C>, matrix<T, R, C>>\
    FUNC_NAME(std::tuple<T L_MODIFICATION,\
                         T L_MODIFICATION \
                         > lhs,\
              std::tuple<matrix<T, R, C> M_MODIFICATION,\
                         matrix<T, R, C> M_MODIFICATION \
                         > mid,\
              std::tuple<matrix<T, R, C> R_MODIFICATION,\
                         matrix<T, R, C> R_MODIFICATION \
                         > rhs) noexcept\
    {\
        return FUNC_NAME(\
            std::tuple<T, T>(std::get<0>(lhs), std::get<1>(lhs)),\
            std::tuple<matrix<T, R, C> const&, matrix<T, R, C> const&>(\
                std::get<0>(mid), std::get<1>(mid)),                         \
            std::tuple<matrix<T, R, C> const&, matrix<T, R, C> const&>(\
                std::get<0>(rhs), std::get<1>(rhs)));                        \
    }\
    template<typename T, std::size_t R, std::size_t C>\
    MAVE_INLINE std::tuple<matrix<T, R, C>, matrix<T, R, C>, matrix<T, R, C>>\
    FUNC_NAME(std::tuple<T L_MODIFICATION,\
                         T L_MODIFICATION,\
                         T L_MODIFICATION \
                         > lhs,\
              std::tuple<matrix<T, R, C> M_MODIFICATION,\
                         matrix<T, R, C> M_MODIFICATION,\
                         matrix<T, R, C> M_MODIFICATION \
                         > mid,\
              std::tuple<matrix<T, R, C> R_MODIFICATION,\
                         matrix<T, R, C> R_MODIFICATION,\
                         matrix<T, R, C> R_MODIFICATION \
                         > rhs) noexcept\
    {\
        return FUNC_NAME(\
            std::tuple<T, T, T>(std::get<0>(lhs), std::get<1>(lhs), std::get<2>(lhs)),\
            std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&,\
                       const matrix<T, R, C>&>(                       \
                std::get<0>(mid), std::get<1>(mid), std::get<2>(mid)),   \
            std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&,\
                       const matrix<T, R, C>&>(                       \
                std::get<0>(rhs), std::get<1>(rhs), std::get<2>(rhs)));  \
    }\
    template<typename T, std::size_t R, std::size_t C>\
    MAVE_INLINE std::tuple<matrix<T, R, C>, matrix<T, R, C>,\
                           matrix<T, R, C>, matrix<T, R, C>>\
    FUNC_NAME(std::tuple<T L_MODIFICATION,\
                         T L_MODIFICATION,\
                         T L_MODIFICATION,\
                         T L_MODIFICATION \
                         > lhs,\
              std::tuple<matrix<T, R, C> M_MODIFICATION,\
                         matrix<T, R, C> M_MODIFICATION,\
                         matrix<T, R, C> M_MODIFICATION,\
                         matrix<T, R, C> M_MODIFICATION \
                         > mid,\
              std::tuple<matrix<T, R, C> R_MODIFICATION,\
                         matrix<T, R, C> R_MODIFICATION,\
                         matrix<T, R, C> R_MODIFICATION,\
                         matrix<T, R, C> R_MODIFICATION \
                         > rhs) noexcept\
    {\
        return FUNC_NAME(\
            std::tuple<T, T, T>(std::get<0>(lhs), std::get<1>(lhs), \
                                std::get<2>(lhs), std::get<3>(lhs)),\
            std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&, \
                       const matrix<T, R, C>&, const matrix<T, R, C>&>(\
            std::get<0>(mid), std::get<1>(mid), std::get<2>(mid), std::get<3>(mid)),\
            std::tuple<const matrix<T, R, C>&, const matrix<T, R, C>&, \
                       const matrix<T, R, C>&, const matrix<T, R, C>&>(\
            std::get<0>(rhs), std::get<1>(rhs), std::get<2>(rhs), std::get<3>(rhs)));\
    }\
    /**/

MAVE_GENERATE_FORWARDING_TERNARY_FUNCTIONS_MATRIX_SCALAR_MATRIX_MATRIX(fmadd,  MAVE_EMPTY_ARGUMENT, MAVE_EMPTY_ARGUMENT, MAVE_EMPTY_ARGUMENT)
MAVE_GENERATE_FORWARDING_TERNARY_FUNCTIONS_MATRIX_SCALAR_MATRIX_MATRIX(fmadd,  const&             , MAVE_EMPTY_ARGUMENT, MAVE_EMPTY_ARGUMENT)
MAVE_GENERATE_FORWARDING_TERNARY_FUNCTIONS_MATRIX_SCALAR_MATRIX_MATRIX(fmadd,  &                  , MAVE_EMPTY_ARGUMENT, MAVE_EMPTY_ARGUMENT)
MAVE_GENERATE_FORWARDING_TERNARY_FUNCTIONS_MATRIX_SCALAR_MATRIX_MATRIX(fmadd,  MAVE_EMPTY_ARGUMENT, &,                   MAVE_EMPTY_ARGUMENT)
MAVE_GENERATE_FORWARDING_TERNARY_FUNCTIONS_MATRIX_SCALAR_MATRIX_MATRIX(fmadd,  const&             , &,                   MAVE_EMPTY_ARGUMENT)
MAVE_GENERATE_FORWARDING_TERNARY_FUNCTIONS_MATRIX_SCALAR_MATRIX_MATRIX(fmadd,  &                  , &,                   MAVE_EMPTY_ARGUMENT)
MAVE_GENERATE_FORWARDING_TERNARY_FUNCTIONS_MATRIX_SCALAR_MATRIX_MATRIX(fmadd,  MAVE_EMPTY_ARGUMENT, const&,              MAVE_EMPTY_ARGUMENT)
MAVE_GENERATE_FORWARDING_TERNARY_FUNCTIONS_MATRIX_SCALAR_MATRIX_MATRIX(fmadd,  const&             , const&,              MAVE_EMPTY_ARGUMENT)
MAVE_GENERATE_FORWARDING_TERNARY_FUNCTIONS_MATRIX_SCALAR_MATRIX_MATRIX(fmadd,  &                  , const&,              MAVE_EMPTY_ARGUMENT)
MAVE_GENERATE_FORWARDING_TERNARY_FUNCTIONS_MATRIX_SCALAR_MATRIX_MATRIX(fmadd,  MAVE_EMPTY_ARGUMENT, MAVE_EMPTY_ARGUMENT, &)
MAVE_GENERATE_FORWARDING_TERNARY_FUNCTIONS_MATRIX_SCALAR_MATRIX_MATRIX(fmadd,  const&             , MAVE_EMPTY_ARGUMENT, &)
MAVE_GENERATE_FORWARDING_TERNARY_FUNCTIONS_MATRIX_SCALAR_MATRIX_MATRIX(fmadd,  &                  , MAVE_EMPTY_ARGUMENT, &)
MAVE_GENERATE_FORWARDING_TERNARY_FUNCTIONS_MATRIX_SCALAR_MATRIX_MATRIX(fmadd,  MAVE_EMPTY_ARGUMENT, &,                   &)
MAVE_GENERATE_FORWARDING_TERNARY_FUNCTIONS_MATRIX_SCALAR_MATRIX_MATRIX(fmadd,  const&             , &,                   &)
MAVE_GENERATE_FORWARDING_TERNARY_FUNCTIONS_MATRIX_SCALAR_MATRIX_MATRIX(fmadd,  &                  , &,                   &)
MAVE_GENERATE_FORWARDING_TERNARY_FUNCTIONS_MATRIX_SCALAR_MATRIX_MATRIX(fmadd,  MAVE_EMPTY_ARGUMENT, const&,              &)
MAVE_GENERATE_FORWARDING_TERNARY_FUNCTIONS_MATRIX_SCALAR_MATRIX_MATRIX(fmadd,  const&             , const&,              &)
MAVE_GENERATE_FORWARDING_TERNARY_FUNCTIONS_MATRIX_SCALAR_MATRIX_MATRIX(fmadd,  &                  , const&,              &)
MAVE_GENERATE_FORWARDING_TERNARY_FUNCTIONS_MATRIX_SCALAR_MATRIX_MATRIX(fmadd,  MAVE_EMPTY_ARGUMENT, MAVE_EMPTY_ARGUMENT, const&)
MAVE_GENERATE_FORWARDING_TERNARY_FUNCTIONS_MATRIX_SCALAR_MATRIX_MATRIX(fmadd,  const&             , MAVE_EMPTY_ARGUMENT, const&)
MAVE_GENERATE_FORWARDING_TERNARY_FUNCTIONS_MATRIX_SCALAR_MATRIX_MATRIX(fmadd,  &                  , MAVE_EMPTY_ARGUMENT, const&)
MAVE_GENERATE_FORWARDING_TERNARY_FUNCTIONS_MATRIX_SCALAR_MATRIX_MATRIX(fmadd,  MAVE_EMPTY_ARGUMENT, &,                   const&)
MAVE_GENERATE_FORWARDING_TERNARY_FUNCTIONS_MATRIX_SCALAR_MATRIX_MATRIX(fmadd,  const&             , &,                   const&)
MAVE_GENERATE_FORWARDING_TERNARY_FUNCTIONS_MATRIX_SCALAR_MATRIX_MATRIX(fmadd,  &                  , &,                   const&)
MAVE_GENERATE_FORWARDING_TERNARY_FUNCTIONS_MATRIX_SCALAR_MATRIX_MATRIX(fmadd,  const&             , const&,              const&)
MAVE_GENERATE_FORWARDING_TERNARY_FUNCTIONS_MATRIX_SCALAR_MATRIX_MATRIX(fmadd,  &                  , const&,              const&)

MAVE_GENERATE_FORWARDING_TERNARY_FUNCTIONS_MATRIX_SCALAR_MATRIX_MATRIX(fmsub,  MAVE_EMPTY_ARGUMENT, MAVE_EMPTY_ARGUMENT, MAVE_EMPTY_ARGUMENT)
MAVE_GENERATE_FORWARDING_TERNARY_FUNCTIONS_MATRIX_SCALAR_MATRIX_MATRIX(fmsub,  const&             , MAVE_EMPTY_ARGUMENT, MAVE_EMPTY_ARGUMENT)
MAVE_GENERATE_FORWARDING_TERNARY_FUNCTIONS_MATRIX_SCALAR_MATRIX_MATRIX(fmsub,  &                  , MAVE_EMPTY_ARGUMENT, MAVE_EMPTY_ARGUMENT)
MAVE_GENERATE_FORWARDING_TERNARY_FUNCTIONS_MATRIX_SCALAR_MATRIX_MATRIX(fmsub,  MAVE_EMPTY_ARGUMENT, &,                   MAVE_EMPTY_ARGUMENT)
MAVE_GENERATE_FORWARDING_TERNARY_FUNCTIONS_MATRIX_SCALAR_MATRIX_MATRIX(fmsub,  const&             , &,                   MAVE_EMPTY_ARGUMENT)
MAVE_GENERATE_FORWARDING_TERNARY_FUNCTIONS_MATRIX_SCALAR_MATRIX_MATRIX(fmsub,  &                  , &,                   MAVE_EMPTY_ARGUMENT)
MAVE_GENERATE_FORWARDING_TERNARY_FUNCTIONS_MATRIX_SCALAR_MATRIX_MATRIX(fmsub,  MAVE_EMPTY_ARGUMENT, const&,              MAVE_EMPTY_ARGUMENT)
MAVE_GENERATE_FORWARDING_TERNARY_FUNCTIONS_MATRIX_SCALAR_MATRIX_MATRIX(fmsub,  const&             , const&,              MAVE_EMPTY_ARGUMENT)
MAVE_GENERATE_FORWARDING_TERNARY_FUNCTIONS_MATRIX_SCALAR_MATRIX_MATRIX(fmsub,  &                  , const&,              MAVE_EMPTY_ARGUMENT)
MAVE_GENERATE_FORWARDING_TERNARY_FUNCTIONS_MATRIX_SCALAR_MATRIX_MATRIX(fmsub,  MAVE_EMPTY_ARGUMENT, MAVE_EMPTY_ARGUMENT, &)
MAVE_GENERATE_FORWARDING_TERNARY_FUNCTIONS_MATRIX_SCALAR_MATRIX_MATRIX(fmsub,  const&             , MAVE_EMPTY_ARGUMENT, &)
MAVE_GENERATE_FORWARDING_TERNARY_FUNCTIONS_MATRIX_SCALAR_MATRIX_MATRIX(fmsub,  &                  , MAVE_EMPTY_ARGUMENT, &)
MAVE_GENERATE_FORWARDING_TERNARY_FUNCTIONS_MATRIX_SCALAR_MATRIX_MATRIX(fmsub,  MAVE_EMPTY_ARGUMENT, &,                   &)
MAVE_GENERATE_FORWARDING_TERNARY_FUNCTIONS_MATRIX_SCALAR_MATRIX_MATRIX(fmsub,  const&             , &,                   &)
MAVE_GENERATE_FORWARDING_TERNARY_FUNCTIONS_MATRIX_SCALAR_MATRIX_MATRIX(fmsub,  &                  , &,                   &)
MAVE_GENERATE_FORWARDING_TERNARY_FUNCTIONS_MATRIX_SCALAR_MATRIX_MATRIX(fmsub,  MAVE_EMPTY_ARGUMENT, const&,              &)
MAVE_GENERATE_FORWARDING_TERNARY_FUNCTIONS_MATRIX_SCALAR_MATRIX_MATRIX(fmsub,  const&             , const&,              &)
MAVE_GENERATE_FORWARDING_TERNARY_FUNCTIONS_MATRIX_SCALAR_MATRIX_MATRIX(fmsub,  &                  , const&,              &)
MAVE_GENERATE_FORWARDING_TERNARY_FUNCTIONS_MATRIX_SCALAR_MATRIX_MATRIX(fmsub,  MAVE_EMPTY_ARGUMENT, MAVE_EMPTY_ARGUMENT, const&)
MAVE_GENERATE_FORWARDING_TERNARY_FUNCTIONS_MATRIX_SCALAR_MATRIX_MATRIX(fmsub,  const&             , MAVE_EMPTY_ARGUMENT, const&)
MAVE_GENERATE_FORWARDING_TERNARY_FUNCTIONS_MATRIX_SCALAR_MATRIX_MATRIX(fmsub,  &                  , MAVE_EMPTY_ARGUMENT, const&)
MAVE_GENERATE_FORWARDING_TERNARY_FUNCTIONS_MATRIX_SCALAR_MATRIX_MATRIX(fmsub,  MAVE_EMPTY_ARGUMENT, &,                   const&)
MAVE_GENERATE_FORWARDING_TERNARY_FUNCTIONS_MATRIX_SCALAR_MATRIX_MATRIX(fmsub,  const&             , &,                   const&)
MAVE_GENERATE_FORWARDING_TERNARY_FUNCTIONS_MATRIX_SCALAR_MATRIX_MATRIX(fmsub,  &                  , &,                   const&)
MAVE_GENERATE_FORWARDING_TERNARY_FUNCTIONS_MATRIX_SCALAR_MATRIX_MATRIX(fmsub,  const&             , const&,              const&)
MAVE_GENERATE_FORWARDING_TERNARY_FUNCTIONS_MATRIX_SCALAR_MATRIX_MATRIX(fmsub,  &                  , const&,              const&)

MAVE_GENERATE_FORWARDING_TERNARY_FUNCTIONS_MATRIX_SCALAR_MATRIX_MATRIX(fnmadd, MAVE_EMPTY_ARGUMENT, MAVE_EMPTY_ARGUMENT, MAVE_EMPTY_ARGUMENT)
MAVE_GENERATE_FORWARDING_TERNARY_FUNCTIONS_MATRIX_SCALAR_MATRIX_MATRIX(fnmadd, const&             , MAVE_EMPTY_ARGUMENT, MAVE_EMPTY_ARGUMENT)
MAVE_GENERATE_FORWARDING_TERNARY_FUNCTIONS_MATRIX_SCALAR_MATRIX_MATRIX(fnmadd, &                  , MAVE_EMPTY_ARGUMENT, MAVE_EMPTY_ARGUMENT)
MAVE_GENERATE_FORWARDING_TERNARY_FUNCTIONS_MATRIX_SCALAR_MATRIX_MATRIX(fnmadd, MAVE_EMPTY_ARGUMENT, &,                   MAVE_EMPTY_ARGUMENT)
MAVE_GENERATE_FORWARDING_TERNARY_FUNCTIONS_MATRIX_SCALAR_MATRIX_MATRIX(fnmadd, const&             , &,                   MAVE_EMPTY_ARGUMENT)
MAVE_GENERATE_FORWARDING_TERNARY_FUNCTIONS_MATRIX_SCALAR_MATRIX_MATRIX(fnmadd, &                  , &,                   MAVE_EMPTY_ARGUMENT)
MAVE_GENERATE_FORWARDING_TERNARY_FUNCTIONS_MATRIX_SCALAR_MATRIX_MATRIX(fnmadd, MAVE_EMPTY_ARGUMENT, const&,              MAVE_EMPTY_ARGUMENT)
MAVE_GENERATE_FORWARDING_TERNARY_FUNCTIONS_MATRIX_SCALAR_MATRIX_MATRIX(fnmadd, const&             , const&,              MAVE_EMPTY_ARGUMENT)
MAVE_GENERATE_FORWARDING_TERNARY_FUNCTIONS_MATRIX_SCALAR_MATRIX_MATRIX(fnmadd, &                  , const&,              MAVE_EMPTY_ARGUMENT)
MAVE_GENERATE_FORWARDING_TERNARY_FUNCTIONS_MATRIX_SCALAR_MATRIX_MATRIX(fnmadd, MAVE_EMPTY_ARGUMENT, MAVE_EMPTY_ARGUMENT, &)
MAVE_GENERATE_FORWARDING_TERNARY_FUNCTIONS_MATRIX_SCALAR_MATRIX_MATRIX(fnmadd, const&             , MAVE_EMPTY_ARGUMENT, &)
MAVE_GENERATE_FORWARDING_TERNARY_FUNCTIONS_MATRIX_SCALAR_MATRIX_MATRIX(fnmadd, &                  , MAVE_EMPTY_ARGUMENT, &)
MAVE_GENERATE_FORWARDING_TERNARY_FUNCTIONS_MATRIX_SCALAR_MATRIX_MATRIX(fnmadd, MAVE_EMPTY_ARGUMENT, &,                   &)
MAVE_GENERATE_FORWARDING_TERNARY_FUNCTIONS_MATRIX_SCALAR_MATRIX_MATRIX(fnmadd, const&             , &,                   &)
MAVE_GENERATE_FORWARDING_TERNARY_FUNCTIONS_MATRIX_SCALAR_MATRIX_MATRIX(fnmadd, &                  , &,                   &)
MAVE_GENERATE_FORWARDING_TERNARY_FUNCTIONS_MATRIX_SCALAR_MATRIX_MATRIX(fnmadd, MAVE_EMPTY_ARGUMENT, const&,              &)
MAVE_GENERATE_FORWARDING_TERNARY_FUNCTIONS_MATRIX_SCALAR_MATRIX_MATRIX(fnmadd, const&             , const&,              &)
MAVE_GENERATE_FORWARDING_TERNARY_FUNCTIONS_MATRIX_SCALAR_MATRIX_MATRIX(fnmadd, &                  , const&,              &)
MAVE_GENERATE_FORWARDING_TERNARY_FUNCTIONS_MATRIX_SCALAR_MATRIX_MATRIX(fnmadd, MAVE_EMPTY_ARGUMENT, MAVE_EMPTY_ARGUMENT, const&)
MAVE_GENERATE_FORWARDING_TERNARY_FUNCTIONS_MATRIX_SCALAR_MATRIX_MATRIX(fnmadd, const&             , MAVE_EMPTY_ARGUMENT, const&)
MAVE_GENERATE_FORWARDING_TERNARY_FUNCTIONS_MATRIX_SCALAR_MATRIX_MATRIX(fnmadd, &                  , MAVE_EMPTY_ARGUMENT, const&)
MAVE_GENERATE_FORWARDING_TERNARY_FUNCTIONS_MATRIX_SCALAR_MATRIX_MATRIX(fnmadd, MAVE_EMPTY_ARGUMENT, &,                   const&)
MAVE_GENERATE_FORWARDING_TERNARY_FUNCTIONS_MATRIX_SCALAR_MATRIX_MATRIX(fnmadd, const&             , &,                   const&)
MAVE_GENERATE_FORWARDING_TERNARY_FUNCTIONS_MATRIX_SCALAR_MATRIX_MATRIX(fnmadd, &                  , &,                   const&)
MAVE_GENERATE_FORWARDING_TERNARY_FUNCTIONS_MATRIX_SCALAR_MATRIX_MATRIX(fnmadd, const&             , const&,              const&)
MAVE_GENERATE_FORWARDING_TERNARY_FUNCTIONS_MATRIX_SCALAR_MATRIX_MATRIX(fnmadd, &                  , const&,              const&)

MAVE_GENERATE_FORWARDING_TERNARY_FUNCTIONS_MATRIX_SCALAR_MATRIX_MATRIX(fnmsub, MAVE_EMPTY_ARGUMENT, MAVE_EMPTY_ARGUMENT, MAVE_EMPTY_ARGUMENT)
MAVE_GENERATE_FORWARDING_TERNARY_FUNCTIONS_MATRIX_SCALAR_MATRIX_MATRIX(fnmsub, const&             , MAVE_EMPTY_ARGUMENT, MAVE_EMPTY_ARGUMENT)
MAVE_GENERATE_FORWARDING_TERNARY_FUNCTIONS_MATRIX_SCALAR_MATRIX_MATRIX(fnmsub, &                  , MAVE_EMPTY_ARGUMENT, MAVE_EMPTY_ARGUMENT)
MAVE_GENERATE_FORWARDING_TERNARY_FUNCTIONS_MATRIX_SCALAR_MATRIX_MATRIX(fnmsub, MAVE_EMPTY_ARGUMENT, &,                   MAVE_EMPTY_ARGUMENT)
MAVE_GENERATE_FORWARDING_TERNARY_FUNCTIONS_MATRIX_SCALAR_MATRIX_MATRIX(fnmsub, const&             , &,                   MAVE_EMPTY_ARGUMENT)
MAVE_GENERATE_FORWARDING_TERNARY_FUNCTIONS_MATRIX_SCALAR_MATRIX_MATRIX(fnmsub, &                  , &,                   MAVE_EMPTY_ARGUMENT)
MAVE_GENERATE_FORWARDING_TERNARY_FUNCTIONS_MATRIX_SCALAR_MATRIX_MATRIX(fnmsub, MAVE_EMPTY_ARGUMENT, const&,              MAVE_EMPTY_ARGUMENT)
MAVE_GENERATE_FORWARDING_TERNARY_FUNCTIONS_MATRIX_SCALAR_MATRIX_MATRIX(fnmsub, const&             , const&,              MAVE_EMPTY_ARGUMENT)
MAVE_GENERATE_FORWARDING_TERNARY_FUNCTIONS_MATRIX_SCALAR_MATRIX_MATRIX(fnmsub, &                  , const&,              MAVE_EMPTY_ARGUMENT)
MAVE_GENERATE_FORWARDING_TERNARY_FUNCTIONS_MATRIX_SCALAR_MATRIX_MATRIX(fnmsub, MAVE_EMPTY_ARGUMENT, MAVE_EMPTY_ARGUMENT, &)
MAVE_GENERATE_FORWARDING_TERNARY_FUNCTIONS_MATRIX_SCALAR_MATRIX_MATRIX(fnmsub, const&             , MAVE_EMPTY_ARGUMENT, &)
MAVE_GENERATE_FORWARDING_TERNARY_FUNCTIONS_MATRIX_SCALAR_MATRIX_MATRIX(fnmsub, &                  , MAVE_EMPTY_ARGUMENT, &)
MAVE_GENERATE_FORWARDING_TERNARY_FUNCTIONS_MATRIX_SCALAR_MATRIX_MATRIX(fnmsub, MAVE_EMPTY_ARGUMENT, &,                   &)
MAVE_GENERATE_FORWARDING_TERNARY_FUNCTIONS_MATRIX_SCALAR_MATRIX_MATRIX(fnmsub, const&             , &,                   &)
MAVE_GENERATE_FORWARDING_TERNARY_FUNCTIONS_MATRIX_SCALAR_MATRIX_MATRIX(fnmsub, &                  , &,                   &)
MAVE_GENERATE_FORWARDING_TERNARY_FUNCTIONS_MATRIX_SCALAR_MATRIX_MATRIX(fnmsub, MAVE_EMPTY_ARGUMENT, const&,              &)
MAVE_GENERATE_FORWARDING_TERNARY_FUNCTIONS_MATRIX_SCALAR_MATRIX_MATRIX(fnmsub, const&             , const&,              &)
MAVE_GENERATE_FORWARDING_TERNARY_FUNCTIONS_MATRIX_SCALAR_MATRIX_MATRIX(fnmsub, &                  , const&,              &)
MAVE_GENERATE_FORWARDING_TERNARY_FUNCTIONS_MATRIX_SCALAR_MATRIX_MATRIX(fnmsub, MAVE_EMPTY_ARGUMENT, MAVE_EMPTY_ARGUMENT, const&)
MAVE_GENERATE_FORWARDING_TERNARY_FUNCTIONS_MATRIX_SCALAR_MATRIX_MATRIX(fnmsub, const&             , MAVE_EMPTY_ARGUMENT, const&)
MAVE_GENERATE_FORWARDING_TERNARY_FUNCTIONS_MATRIX_SCALAR_MATRIX_MATRIX(fnmsub, &                  , MAVE_EMPTY_ARGUMENT, const&)
MAVE_GENERATE_FORWARDING_TERNARY_FUNCTIONS_MATRIX_SCALAR_MATRIX_MATRIX(fnmsub, MAVE_EMPTY_ARGUMENT, &,                   const&)
MAVE_GENERATE_FORWARDING_TERNARY_FUNCTIONS_MATRIX_SCALAR_MATRIX_MATRIX(fnmsub, const&             , &,                   const&)
MAVE_GENERATE_FORWARDING_TERNARY_FUNCTIONS_MATRIX_SCALAR_MATRIX_MATRIX(fnmsub, &                  , &,                   const&)
MAVE_GENERATE_FORWARDING_TERNARY_FUNCTIONS_MATRIX_SCALAR_MATRIX_MATRIX(fnmsub, const&             , const&,              const&)
MAVE_GENERATE_FORWARDING_TERNARY_FUNCTIONS_MATRIX_SCALAR_MATRIX_MATRIX(fnmsub, &                  , const&,              const&)
#undef MAVE_GENERATE_FORWARDING_TERNARY_FUNCTIONS_MATRIX_SCALAR_MATRIX_MATRIX


// -------------------------------------------------------------------------
// for vector<T, 3>: length, length_sq, rlength, regularize, dot_product, cross_product
// -------------------------------------------------------------------------

#undef  MAVE_GENERATE_EXPANDING_UNARY_FUNCTIONS_SCALAR_VECTOR3
#define MAVE_GENERATE_EXPANDING_UNARY_FUNCTIONS_SCALAR_VECTOR3(FUNC_NAME, MODIFICATION)\
    template<typename T>\
    MAVE_INLINE std::pair<T, T>\
    FUNC_NAME(std::tuple<matrix<T, 3, 1> MODIFICATION,\
                         matrix<T, 3, 1> MODIFICATION \
                         > ms) noexcept\
    {\
        return FUNC_NAME(std::get<0>(ms), std::get<1>(ms));\
    }\
    template<typename T>\
    MAVE_INLINE std::pair<T, T>\
    FUNC_NAME(std::pair<matrix<T, 3, 1> MODIFICATION,\
                        matrix<T, 3, 1> MODIFICATION \
                        > ms) noexcept\
    {\
        return FUNC_NAME(std::get<0>(ms), std::get<1>(ms));\
    }\
    template<typename T>\
    MAVE_INLINE std::tuple<T, T, T>\
    FUNC_NAME(std::tuple<matrix<T, 3, 1> MODIFICATION,\
                         matrix<T, 3, 1> MODIFICATION,\
                         matrix<T, 3, 1> MODIFICATION \
                         > ms) noexcept\
    {\
        return FUNC_NAME(std::get<0>(ms), std::get<1>(ms), std::get<2>(ms));\
    }\
    template<typename T>\
    MAVE_INLINE std::tuple<T, T, T, T>\
    FUNC_NAME(std::tuple<matrix<T, 3, 1> MODIFICATION,\
                         matrix<T, 3, 1> MODIFICATION,\
                         matrix<T, 3, 1> MODIFICATION,\
                         matrix<T, 3, 1> MODIFICATION \
                         > ms) noexcept\
    {\
        return FUNC_NAME(std::get<0>(ms), std::get<1>(ms),\
                         std::get<2>(ms), std::get<3>(ms));\
    }\
    /**/

MAVE_GENERATE_EXPANDING_UNARY_FUNCTIONS_SCALAR_VECTOR3(length_sq, MAVE_EMPTY_ARGUMENT)
MAVE_GENERATE_EXPANDING_UNARY_FUNCTIONS_SCALAR_VECTOR3(length_sq, const &)
MAVE_GENERATE_EXPANDING_UNARY_FUNCTIONS_SCALAR_VECTOR3(length_sq, &)
MAVE_GENERATE_EXPANDING_UNARY_FUNCTIONS_SCALAR_VECTOR3(length,    MAVE_EMPTY_ARGUMENT)
MAVE_GENERATE_EXPANDING_UNARY_FUNCTIONS_SCALAR_VECTOR3(length,    const&)
MAVE_GENERATE_EXPANDING_UNARY_FUNCTIONS_SCALAR_VECTOR3(length,    &)
MAVE_GENERATE_EXPANDING_UNARY_FUNCTIONS_SCALAR_VECTOR3(rlength,   MAVE_EMPTY_ARGUMENT)
MAVE_GENERATE_EXPANDING_UNARY_FUNCTIONS_SCALAR_VECTOR3(rlength,   const&)
MAVE_GENERATE_EXPANDING_UNARY_FUNCTIONS_SCALAR_VECTOR3(rlength,   &)
#undef MAVE_GENERATE_EXPANDING_UNARY_FUNCTIONS_SCALAR_VECTOR3


#undef  MAVE_GENERATE_FORWARDING_REGULARIZE
#define MAVE_GENERATE_FORWARDING_REGULARIZE(MODIFICATION)\
    template<typename T>\
    MAVE_INLINE std::pair<std::pair<vector<T, 3>, T>, std::pair<vector<T, 3>, T>>\
    regularize(std::tuple<matrix<T, 3, 1> MODIFICATION,\
                          matrix<T, 3, 1> MODIFICATION \
                          > ms) noexcept\
    {\
        return FUNC_NAME(\
            std::tuple<const matrix<T, 3, 1>&, const matrix<T, 3, 1>&>(\
                std::get<0>(ms), std::get<1>(ms)));\
    }\
    template<typename T>\
    MAVE_INLINE std::tuple<std::pair<vector<T, 3>, T>, std::pair<vector<T, 3>, T>,\
                           std::pair<vector<T, 3>, T>>\
    regularize(std::tuple<matrix<T, 3, 1> MODIFICATION,\
                          matrix<T, 3, 1> MODIFICATION,\
                          matrix<T, 3, 1> MODIFICATION \
                          > ms) noexcept\
    {\
        return FUNC_NAME(\
            std::tuple<const matrix<T, 3, 1>&, const matrix<T, 3, 1>&,\
                       const matrix<T, 3, 1>&>(\
                std::get<0>(ms), std::get<1>(ms), std::get<2>(ms)));\
    }\
    template<typename T>\
    MAVE_INLINE std::tuple<std::pair<vector<T, 3>, T>, std::pair<vector<T, 3>, T>,\
                           std::pair<vector<T, 3>, T>, std::pair<vector<T, 3>, T>>\
    regularize(std::tuple<matrix<T, 3, 1> MODIFICATION,\
                          matrix<T, 3, 1> MODIFICATION,\
                          matrix<T, 3, 1> MODIFICATION,\
                          matrix<T, 3, 1> MODIFICATION \
                          > ms) noexcept\
    {\
        return FUNC_NAME(\
            std::tuple<const matrix<T, 3, 1>&, const matrix<T, 3, 1>&,\
                       const matrix<T, 3, 1>&, const matrix<T, 3, 1>&>(\
            std::get<0>(ms), std::get<1>(ms), std::get<2>(ms), std::get<3>(ms)));\
    }\
    /**/

MAVE_GENERATE_FORWARDING_REGULARIZE(MAVE_EMPTY_ARGUMENT)
MAVE_GENERATE_FORWARDING_REGULARIZE(&)
#undef MAVE_GENERATE_FORWARDING_REGULARIZE


#undef  MAVE_GENERATE_FORWARDING_BINARY_FUNCTIONS_SCALAR_VECTOR3_VECTOR3
#define MAVE_GENERATE_FORWARDING_BINARY_FUNCTIONS_SCALAR_VECTOR3_VECTOR3(\
        FUNC_NAME, L_MODIFICATION, R_MODIFICATION)\
    template<typename T>\
    MAVE_INLINE std::pair<T, T>\
    FUNC_NAME(std::tuple<matrix<T, 3, 1> L_MODIFICATION,\
                         matrix<T, 3, 1> L_MODIFICATION \
                         > lhs,\
              std::tuple<matrix<T, 3, 1> R_MODIFICATION,\
                         matrix<T, 3, 1> R_MODIFICATION \
                         > rhs) noexcept\
    {\
        return FUNC_NAME(\
            std::tuple<const matrix<T, 3, 1>&, const matrix<T, 3, 1>&>(\
                std::get<0>(lhs), std::get<1>(lhs)),\
            std::tuple<const matrix<T, 3, 1>&, const matrix<T, 3, 1>&>(\
                std::get<0>(rhs), std::get<1>(rhs)));\
    }\
    template<typename T>\
    MAVE_INLINE std::tuple<T, T, T>\
    FUNC_NAME(std::tuple<matrix<T, 3, 1> L_MODIFICATION,\
                         matrix<T, 3, 1> L_MODIFICATION,\
                         matrix<T, 3, 1> L_MODIFICATION \
                         > lhs,\
              std::tuple<matrix<T, 3, 1> R_MODIFICATION,\
                         matrix<T, 3, 1> R_MODIFICATION,\
                         matrix<T, 3, 1> R_MODIFICATION \
                         > rhs) noexcept\
    {\
        return FUNC_NAME(\
            std::tuple<const matrix<T, 3, 1>&, const matrix<T, 3, 1>&,\
                       const matrix<T, 3, 1>&>(\
                std::get<0>(lhs), std::get<1>(lhs), std::get<2>(lhs)),\
            std::tuple<const matrix<T, 3, 1>&, const matrix<T, 3, 1>&,\
                       const matrix<T, 3, 1>&>(\
                std::get<0>(rhs), std::get<1>(rhs), std::get<2>(rhs)));\
    }\
    template<typename T>\
    MAVE_INLINE std::tuple<T, T, T, T>\
    FUNC_NAME(std::tuple<matrix<T, 3, 1> L_MODIFICATION,\
                         matrix<T, 3, 1> L_MODIFICATION,\
                         matrix<T, 3, 1> L_MODIFICATION,\
                         matrix<T, 3, 1> L_MODIFICATION \
                         > lhs,\
              std::tuple<matrix<T, 3, 1> R_MODIFICATION,\
                         matrix<T, 3, 1> R_MODIFICATION,\
                         matrix<T, 3, 1> R_MODIFICATION,\
                         matrix<T, 3, 1> R_MODIFICATION \
                         > rhs) noexcept\
    {\
        return FUNC_NAME(\
            std::tuple<const matrix<T, 3, 1>&, const matrix<T, 3, 1>&,\
                       const matrix<T, 3, 1>&, const matrix<T, 3, 1>&>(\
            std::get<0>(lhs), std::get<1>(lhs), std::get<2>(lhs), std::get<3>(lhs)),\
            std::tuple<const matrix<T, 3, 1>&, const matrix<T, 3, 1>&,\
                       const matrix<T, 3, 1>&, const matrix<T, 3, 1>&>(\
            std::get<0>(rhs), std::get<1>(rhs), std::get<2>(rhs), std::get<3>(rhs)));\
    }\
    /**/

MAVE_GENERATE_FORWARDING_BINARY_FUNCTIONS_SCALAR_VECTOR3_VECTOR3(dot_product, MAVE_EMPTY_ARGUMENT, MAVE_EMPTY_ARGUMENT)
MAVE_GENERATE_FORWARDING_BINARY_FUNCTIONS_SCALAR_VECTOR3_VECTOR3(dot_product, const&,              MAVE_EMPTY_ARGUMENT)
MAVE_GENERATE_FORWARDING_BINARY_FUNCTIONS_SCALAR_VECTOR3_VECTOR3(dot_product, &,                   MAVE_EMPTY_ARGUMENT)
MAVE_GENERATE_FORWARDING_BINARY_FUNCTIONS_SCALAR_VECTOR3_VECTOR3(dot_product, MAVE_EMPTY_ARGUMENT, const&)
MAVE_GENERATE_FORWARDING_BINARY_FUNCTIONS_SCALAR_VECTOR3_VECTOR3(dot_product, &,                   const&)
MAVE_GENERATE_FORWARDING_BINARY_FUNCTIONS_SCALAR_VECTOR3_VECTOR3(dot_product, MAVE_EMPTY_ARGUMENT, &)
MAVE_GENERATE_FORWARDING_BINARY_FUNCTIONS_SCALAR_VECTOR3_VECTOR3(dot_product, const&,              &)
MAVE_GENERATE_FORWARDING_BINARY_FUNCTIONS_SCALAR_VECTOR3_VECTOR3(dot_product, &,                   &)
#undef MAVE_GENERATE_FORWARDING_BINARY_FUNCTIONS_SCALAR_VECTOR3_VECTOR3

#undef  MAVE_GENERATE_FORWARDING_BINARY_FUNCTIONS_VECTOR3_VECTOR3_VECTOR3
#define MAVE_GENERATE_FORWARDING_BINARY_FUNCTIONS_VECTOR3_VECTOR3_VECTOR3(\
        FUNC_NAME, L_MODIFICATION, R_MODIFICATION)\
    template<typename T>\
    MAVE_INLINE std::pair<matrix<T, 3, 1>, matrix<T, 3, 1>>\
    FUNC_NAME(std::tuple<matrix<T, 3, 1> L_MODIFICATION,\
                         matrix<T, 3, 1> L_MODIFICATION \
                         > lhs,\
              std::tuple<matrix<T, 3, 1> R_MODIFICATION,\
                         matrix<T, 3, 1> R_MODIFICATION \
                         > rhs) noexcept\
    {\
        return FUNC_NAME(\
            std::tuple<const matrix<T, 3, 1>&, const matrix<T, 3, 1>&>(\
                std::get<0>(lhs), std::get<1>(lhs)),\
            std::tuple<const matrix<T, 3, 1>&, const matrix<T, 3, 1>&>(\
                std::get<0>(rhs), std::get<1>(rhs)));\
    }\
    template<typename T>\
    MAVE_INLINE std::tuple<matrix<T, 3, 1>, matrix<T, 3, 1>, matrix<T, 3, 1>>\
    FUNC_NAME(std::tuple<matrix<T, 3, 1> L_MODIFICATION,\
                         matrix<T, 3, 1> L_MODIFICATION,\
                         matrix<T, 3, 1> L_MODIFICATION \
                         > lhs,\
              std::tuple<matrix<T, 3, 1> R_MODIFICATION,\
                         matrix<T, 3, 1> R_MODIFICATION,\
                         matrix<T, 3, 1> R_MODIFICATION \
                         > rhs) noexcept\
    {\
        return FUNC_NAME(\
            std::tuple<const matrix<T, 3, 1>&, const matrix<T, 3, 1>&,\
                       const matrix<T, 3, 1>&>(\
                std::get<0>(lhs), std::get<1>(lhs), std::get<2>(lhs)),\
            std::tuple<const matrix<T, 3, 1>&, const matrix<T, 3, 1>&,\
                       const matrix<T, 3, 1>&>(\
                std::get<0>(rhs), std::get<1>(rhs), std::get<2>(rhs)));\
    }\
    template<typename T>\
    MAVE_INLINE std::tuple<matrix<T, 3, 1>, matrix<T, 3, 1>,\
                           matrix<T, 3, 1>, matrix<T, 3, 1>>\
    FUNC_NAME(std::tuple<matrix<T, 3, 1> L_MODIFICATION,\
                         matrix<T, 3, 1> L_MODIFICATION,\
                         matrix<T, 3, 1> L_MODIFICATION,\
                         matrix<T, 3, 1> L_MODIFICATION \
                         > lhs,\
              std::tuple<matrix<T, 3, 1> R_MODIFICATION,\
                         matrix<T, 3, 1> R_MODIFICATION,\
                         matrix<T, 3, 1> R_MODIFICATION,\
                         matrix<T, 3, 1> R_MODIFICATION \
                         > rhs) noexcept\
    {\
        return FUNC_NAME(\
            std::tuple<const matrix<T, 3, 1>&, const matrix<T, 3, 1>&,\
                       const matrix<T, 3, 1>&, const matrix<T, 3, 1>&>(\
            std::get<0>(lhs), std::get<1>(lhs), std::get<2>(lhs), std::get<3>(lhs)),\
            std::tuple<const matrix<T, 3, 1>&, const matrix<T, 3, 1>&,\
                       const matrix<T, 3, 1>&, const matrix<T, 3, 1>&>(\
            std::get<0>(rhs), std::get<1>(rhs), std::get<2>(rhs), std::get<3>(rhs)));\
    }\
    /**/

MAVE_GENERATE_FORWARDING_BINARY_FUNCTIONS_VECTOR3_VECTOR3_VECTOR3(cross_product, MAVE_EMPTY_ARGUMENT, MAVE_EMPTY_ARGUMENT)
MAVE_GENERATE_FORWARDING_BINARY_FUNCTIONS_VECTOR3_VECTOR3_VECTOR3(cross_product, const&,              MAVE_EMPTY_ARGUMENT)
MAVE_GENERATE_FORWARDING_BINARY_FUNCTIONS_VECTOR3_VECTOR3_VECTOR3(cross_product, &,                   MAVE_EMPTY_ARGUMENT)
MAVE_GENERATE_FORWARDING_BINARY_FUNCTIONS_VECTOR3_VECTOR3_VECTOR3(cross_product, MAVE_EMPTY_ARGUMENT, const&)
MAVE_GENERATE_FORWARDING_BINARY_FUNCTIONS_VECTOR3_VECTOR3_VECTOR3(cross_product, &,                   const&)
MAVE_GENERATE_FORWARDING_BINARY_FUNCTIONS_VECTOR3_VECTOR3_VECTOR3(cross_product, MAVE_EMPTY_ARGUMENT, &)
MAVE_GENERATE_FORWARDING_BINARY_FUNCTIONS_VECTOR3_VECTOR3_VECTOR3(cross_product, const&,              &)
MAVE_GENERATE_FORWARDING_BINARY_FUNCTIONS_VECTOR3_VECTOR3_VECTOR3(cross_product, &,                   &)
#undef MAVE_GENERATE_FORWARDING_BINARY_FUNCTIONS_VECTOR3_VECTOR3_VECTOR3

} // mave
#endif// MAVE_FORWARD_HPP
