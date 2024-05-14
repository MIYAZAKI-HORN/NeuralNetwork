
#include "NeuralNetLayerType.h"

void		
DataShape_construct(DataShape* pDataShape, uint32_t	height, uint32_t width, uint32_t channel) {
	if (pDataShape != NULL) {
		pDataShape->height	= height;
		pDataShape->width	= width;
		pDataShape->channel	= channel;
	}
}

void
DataShape_initialize(DataShape* pDataShape) {
	if (pDataShape != NULL) {
		pDataShape->height	= 0;
		pDataShape->width	= 0;
		pDataShape->channel	= 0;
	}
}

uint32_t	
DataShape_getSize(const DataShape* pDataShape) {
	if (pDataShape != NULL) {
		return pDataShape->height * pDataShape->width * pDataShape->channel;
	}
	else {
		return 0;
	}
}

bool_t		
DataShape_equal(const DataShape* pDataShape, const  DataShape* pDataShapeToCompare) {
	if (pDataShape->height != pDataShapeToCompare->height) {
		return FALSE;
	}
	if (pDataShape->width != pDataShapeToCompare->width) {
		return FALSE;
	}
	if (pDataShape->channel != pDataShapeToCompare->channel) {
		return FALSE;
	}
	return TRUE;
}
