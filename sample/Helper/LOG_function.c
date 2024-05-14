#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "LOG_Function.h"

static FILE*	pFile = NULL;
static bool_t	fPrintf = TRUE;

bool_t
OPEN_LOG_FILE(const char* pFileName) {
	pFile = fopen(pFileName, "w");
	if(pFile != NULL) {
		return TRUE;
	} else {
		printf("LOG file open error %s\n",pFileName);
		return FALSE;
	}
}

void	
CLOSE_LOG_FILE(void) {
	if( pFile != NULL ) {
		fclose(pFile);
		pFile = NULL;
	}
}

void
LOG_PRINTF_REQUUIRED(bool_t fRequired) {
	fPrintf = fRequired;
}

void	
SAVE_LOG(const char* pTitle) {
	if (fPrintf == TRUE) {
		printf("%s\n", pTitle);
	}
	if( pFile != NULL ) {
		fwrite(pTitle,1,strlen(pTitle),pFile);
		fwrite("\n",1,1,pFile);
		fflush(pFile);
	}
}

void	
SAVE_LOG_WITHOUT_RETURN(const char* pTitle) {
	if (fPrintf == TRUE) {
		printf("%s", pTitle);
	}
	if( pFile != NULL ) {
		fwrite(pTitle,1,strlen(pTitle),pFile);
		fflush(pFile);
	}
}

void	
SAVE_LOG_WITH_INT(const char* pTitle,int32_t v) {
	char	value[100];
	if (fPrintf == TRUE) {
		printf("%s %ld\n", pTitle, v);
	}
	if( pFile != NULL ) {
		fwrite(pTitle,1,strlen(pTitle),pFile);
		sprintf(value,"%ld",v);
		fwrite(value,1,strlen(value),pFile);
		fwrite("\n",1,1,pFile);
		fflush(pFile);
	}
}

void
SAVE_LOG_WITH_FLT(const char* pTitle,float v) {
	char	value[100];
	if (fPrintf == TRUE) {
		printf("%s %10.3f\n", pTitle, v);
	}
	if( pFile != NULL ) {
		fwrite(pTitle,1,strlen(pTitle),pFile);
		sprintf(value,"%10.3f",v);
		fwrite(value,1,strlen(value),pFile);
		fwrite("\n",1,1,pFile);
		fflush(pFile);
	}
}
void	
SAVE_LOG_WITH_STR(const char* pTitle,const char* pInfo) {
	if (fPrintf == TRUE) {
		printf("%s %s\n", pTitle, pInfo);
	}
	if( pFile != NULL ) {
		fwrite(pTitle,1,strlen(pTitle),pFile);
		fwrite(pInfo,1,strlen(pInfo),pFile);
		fwrite("\n", 1, 1, pFile);
		fflush(pFile);
	}
}

void
SAVE_LOG_FLT_VALUE(float v, const char* pFormat, bool_t fWithReturn) {
	char	value[100];
	if (pFile != NULL) {
		sprintf(value, pFormat, v);
		fwrite(value, 1, strlen(value), pFile);
		if (fWithReturn == TRUE) {
			fwrite("\n", 1, 1, pFile);
		}
		fflush(pFile);
	}
}

