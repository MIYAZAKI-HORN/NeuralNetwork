#ifndef QC_LOG_H
#define QC_LOG_H

#include <stdio.h>
#include <stdlib.h>
#include "STDTypeDefinition.h"

bool_t	OPEN_LOG_FILE(const char* pFileName);
void	CLOSE_LOG_FILE(void);
void	LOG_PRINTF_REQUUIRED(bool_t fRequired);
void	SAVE_LOG(const char* pTitle);
void	SAVE_LOG_WITHOUT_RETURN(const char* pTitle);
void	SAVE_LOG_WITH_INT(const char* pTitle,int32_t v);
void	SAVE_LOG_WITH_FLT(const char* pTitle,float v);
void	SAVE_LOG_WITH_STR(const char* pTitle,const char* pInfo);
void	SAVE_LOG_FLT_VALUE(float v, const char* pFormat, bool_t fWithReturn);

#endif
