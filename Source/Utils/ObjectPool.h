#ifndef TRACKING_OBJECTPOOL_H
#define TRACKING_OBJECTPOOL_H

#include <queue>
#include <vector>

#pragma warning(disable: 4251)

namespace Utils {
	// template class CObjectPool
    template <typename T>
    class CObjectPool
    {
    public:
        // throws invalid_argument if chunkSize <= 0
        CObjectPool(int chunkSize = kDefaultChunkSize) /*throw(std::invalid_argument, std::bad_alloc)*/ : mChunkSize(chunkSize) {};

        // frees all the allocated objects
        ~CObjectPool();

        // reserve an object for use. clients must not free the object!
        T& AcquireObject();
            
        // return object to the pool.
        void ReleaseObject(T& obj);

    protected:
        // mFreeList stores the object that are not currently in use by clients
        std::queue<T*> m_freeObjects;

        // m_allObjects stores points to all the objects, in use or not. to ensure all 
        // objects are freed properly in the destructor.
        std::vector<T*> m_allObjects;

        int mChunkSize;
        static const int kDefaultChunkSize = 10;

        void AllocateChunk();
        static void ArrayDeleteObject(T* obj);

    private:
        CObjectPool<T>(const CObjectPool<T>& src);
        CObjectPool<T>& operator=(const CObjectPool<T>& rhs);

    };

	template <typename T>
	void CObjectPool<T>::AllocateChunk()
	{
		T* newObjects = new T[mChunkSize];
		m_allObjects.push_back(newObjects);
		for (int i = 0; i < mChunkSize; ++i)
		{
			m_freeObjects.push(&newObjects[i]);
		}

	}

	template <typename T>
	void CObjectPool<T>::ArrayDeleteObject(T* obj)
	{
		delete[] obj;
	}


	template <typename T>
	CObjectPool<T>::~CObjectPool()
	{
		for_each(m_allObjects.begin(), m_allObjects.end(), CObjectPool<T>::ArrayDeleteObject);
	}

	template <typename T>
	T& CObjectPool<T>::AcquireObject()
	{
		if (m_freeObjects.empty())
			AllocateChunk();

		T* obj = m_freeObjects.front();
		m_freeObjects.pop();
		return *obj;

	}

	template <typename T>
	void CObjectPool<T>::ReleaseObject(T& obj)
	{
		m_freeObjects.push(& obj);
	}
}

#endif