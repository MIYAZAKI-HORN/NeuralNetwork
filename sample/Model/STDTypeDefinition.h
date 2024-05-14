#ifndef  STANDARD_TYPE_DEFINITION_H
#define  STANDARD_TYPE_DEFINITION_H

#ifdef  __cplusplus
extern "C" {
#endif

// signed 8 bit
#ifndef char_t
    typedef char char_t;
#endif

// object handle
#ifndef handler_t
    typedef void* handle_t;
#endif

// 8 bit signed integer
#ifndef int8_t
    typedef signed char int8_t;
#endif

// 8 bit unsigned integer
#ifndef uint8_t
    typedef unsigned char uint8_t;
#endif

// 16 bit signed integer
#ifndef int16_t
    typedef signed short int int16_t;
#endif

// 16 bit unsigned integer
#ifndef uint16_t
    typedef unsigned short int uint16_t;
#endif

// 32 bit signed integer
#ifndef int32_t
    typedef signed int int32_t;
#endif

// 32 bit unsigned integer
#ifndef uint32_t
    typedef unsigned int uint32_t;
#endif

// 32 bit floating point
#ifndef flt32_t
    typedef float flt32_t;
#endif

// boolean
#ifndef bool_t
    typedef uint32_t bool_t;
#endif

#ifndef  FALSE
    #define FALSE  (0)
#endif

#ifndef  TRUE
    #define TRUE  (1)
#endif

#ifndef  __cplusplus
    #ifndef  NULL
        #define NULL  ((void*)0)
    #endif
#endif

#ifndef size_in_type
    #define  size_in_type(byteSize, UnitType)  (((byteSize) + (sizeof(UnitType) - 1u)) / sizeof(UnitType))
#endif 

#ifndef larger_of
    #define larger_of(x, y)	(((x) >= (y)) ? (x) : (y))
#endif
#ifndef smaller_of
    #define smaller_of(x, y)	(((x) < (y)) ? (x) : (y))
#endif


#ifdef  __cplusplus
}  /* extern "C" */
#endif

#endif
