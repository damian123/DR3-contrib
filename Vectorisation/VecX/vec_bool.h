/****************************  vec_bool.h  *******************************
* Author:        Andrew Drakeford
* Date created:  2021-04-10
* Last modified: 2021-04-10
* Version:       1.0
* Project:       DR Cubed
* Description:
*
* (c) Copyright 2019 Andrew Drakeford
* Apache License version 2.0 or later.
*****************************************************************************/
#pragma once

#include <algorithm>
#include <iterator>
#include <stdexcept>
#include "alloc_policy.h"
#include "apply_operation.h"

template <typename INS_VEC>
class VecBool
{
private:
	static size_t checkedSize(int sz)
	{
		if (sz < 0)
		{
			throw std::invalid_argument("VecBool size must be non-negative");
		}
		return static_cast<size_t>(sz);
	}


	typename InstructionTraits<INS_VEC>::FloatType* m_pData;
	size_t m_size;
	size_t m_implSize;

	bool m_scalarVal;
	bool m_isScalar = false;

public:
	explicit VecBool(bool scalar) : m_pData(nullptr), m_size(0), m_implSize(0),  m_scalarVal(scalar),m_isScalar(true)
	{
		
	}

	explicit VecBool(int sz)
		: m_pData(nullptr), m_size(checkedSize(sz)),
		  m_implSize(m_size), m_scalarVal(false), m_isScalar(false)
	{
		if (sz > 0)
		{
			allocPool(m_implSize, m_pData);
			std::fill_n(m_pData, m_implSize,
				InstructionTraits<INS_VEC>::nullValue);
		}
	}


	~VecBool()
	{
		if (m_pData != nullptr)
		{
			freePool(m_implSize, m_pData);
		}
	}

	VecBool(const VecBool& rhs)
		: m_pData(nullptr), m_size(rhs.m_size), m_implSize(rhs.m_size),
		  m_scalarVal(rhs.m_scalarVal), m_isScalar(rhs.m_isScalar)
	{
		if (!m_isScalar && m_size > 0)
		{
			allocPool(m_implSize, m_pData);
			std::copy(rhs.m_pData, rhs.m_pData + m_implSize, m_pData);
		}
	}


	VecBool& operator=(const VecBool& rhs)
	{
		if (&rhs != this)
		{
			VecBool replacement(rhs);
			swap(replacement);
		}
		return *this;
	}


	VecBool(VecBool&& rhs) noexcept
		: m_pData(rhs.m_pData), m_size(rhs.m_size), m_implSize(rhs.m_implSize),
		  m_scalarVal(rhs.m_scalarVal), m_isScalar(rhs.m_isScalar)
	{
		rhs.m_pData = nullptr;
		rhs.m_size = 0;
		rhs.m_implSize = 0;
		rhs.m_scalarVal = false;
		rhs.m_isScalar = true;
	}

	VecBool& operator=(VecBool&& rhs) noexcept
	{
		if (&rhs != this)
		{
			swap(rhs);
		}
		return *this;
	}

	void swap(VecBool& rhs) noexcept
	{
		using std::swap;
		swap(m_pData, rhs.m_pData);
		swap(m_size, rhs.m_size);
		swap(m_implSize, rhs.m_implSize);
		swap(m_scalarVal, rhs.m_scalarVal);
		swap(m_isScalar, rhs.m_isScalar);
	}


	inline typename InstructionTraits<INS_VEC>::FloatType* start() const
	{
		return m_pData;
	}

	inline size_t size() const
	{
		return m_size;

	}

	inline size_t paddedSize() const
	{
		return m_implSize;
	}

	inline bool getScalarValue() const
	{
		return m_scalarVal;
	}

	inline bool isScalar() const
	{
		return m_isScalar;
	}

	inline bool operator[](int j)const
	{
		return m_pData[j];
	}


	void setAt(int j, bool val)
	{
		m_pData[j] = val; 
	}

};
