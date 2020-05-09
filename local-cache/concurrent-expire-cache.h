//
// Created by chenli on 17/6/13.
// Modified based on HPHP concurrent-lru-cache.h
//

#ifndef ML_PLATFORM_CONCURRENT_EXPIRE_CACHE_H
#define ML_PLATFORM_CONCURRENT_EXPIRE_CACHE_H

#include <atomic>
#include <mutex>
#include <new>
#include <thread>
#include <vector>
#include "tbb/concurrent_hash_map.h"
#include <unordered_map>

namespace ml_platform {

    template <class TKey, class TValue, class THash = tbb::tbb_hash_compare<TKey>>
    struct ConcurrentExpireCache {
    private:
        struct ListNode {
            ListNode() : m_timestamp(0), m_prev(OutOfListMarker), m_next(nullptr)
            {}

            explicit ListNode(const TKey& key, int64_t timestamp)
                    : m_key(key), m_timestamp(timestamp), m_prev(OutOfListMarker), m_next(nullptr)
            {}

            TKey m_key;
            int64_t m_timestamp;
            ListNode* m_prev;
            ListNode* m_next;
        };

        static ListNode* const OutOfListMarker;

        /**
         * The value that we store in the hashtable. The list node is allocated from
         * an internal object_pool. The ListNode* is owned by the list.
         */
        struct HashMapValue {
            HashMapValue()
                    : m_listNode(nullptr)
            {}

            HashMapValue(const TValue& value, ListNode* node)
                    : m_value(value), m_listNode(node)
            {}

            TValue m_value;
            ListNode* m_listNode;
        };

        typedef tbb::concurrent_hash_map<TKey, HashMapValue, THash> HashMap;
        typedef typename HashMap::const_accessor HashMapConstAccessor;
        typedef typename HashMap::accessor HashMapAccessor;
        typedef typename HashMap::value_type HashMapValuePair;

    public:
        /**
         * The proxy object for TBB::CHM::const_accessor. Provides direct access to
         * the user's value by dereferencing, thus hiding our implementation
         * details.
         */
        struct ConstAccessor {
            ConstAccessor() {}

            const TValue& operator*() const {
                return *get();
            }

            const TValue* operator->() const {
                return get();
            }

            const TValue* get() const {
                return &m_hashAccessor->second.m_value;
            }

            const bool will_expire(int sec, int64_t ttlSecond) const {
                std::time_t timestamp = std::time(NULL);
                timestamp -= (ttlSecond - sec);
                ListNode* listNode = m_hashAccessor->second.m_listNode;
                return listNode->m_timestamp < timestamp;
            }

            const void postpone(int sec) const {
                ListNode* listNode = m_hashAccessor->second.m_listNode;
                listNode->m_timestamp += sec;
            }

            bool empty() const {
                return m_hashAccessor.empty();
            }
            
            bool is_valid(void) const {
                ListNode* listNode = m_hashAccessor->second.m_listNode;
                return listNode->m_timestamp > 0;
            }

            void set_invalid(void) const {
                ListNode* listNode = m_hashAccessor->second.m_listNode;
                listNode->m_timestamp = 0;
            }

        private:
            friend struct ConcurrentExpireCache;
            HashMapConstAccessor m_hashAccessor;
        };

        struct Accessor {
            Accessor() {}
            TValue& operator*() const {
                return *get();
            }

            TValue* operator->() const {
                return get();
            }

            TValue* get() const {
                return &m_hashAccessor->second.m_value;
            }

            bool will_expire(int sec, int64_t ttlSecond) const {
                std::time_t timestamp = std::time(NULL);
                timestamp -= (ttlSecond - sec - 10);
                ListNode* listNode = m_hashAccessor->second.m_listNode;
                return listNode->m_timestamp < timestamp;
            }

            void postpone(int sec) const {
                ListNode* listNode = m_hashAccessor->second.m_listNode;
                listNode->m_timestamp += sec;
            }

            bool empty() const {
                return m_hashAccessor.empty();
            }

            bool is_valid(void) const {
                ListNode* listNode = m_hashAccessor->second.m_listNode;
                return listNode->m_timestamp > 0;
            }

            void set_invalid(void) const {
                ListNode* listNode = m_hashAccessor->second.m_listNode;
                listNode->m_timestamp = 0;
            }

        private:
            friend struct ConcurrentExpireCache;
            HashMapAccessor m_hashAccessor;
        };

        explicit ConcurrentExpireCache(size_t maxSize, int64_t ttlSecond);

        ConcurrentExpireCache(const ConcurrentExpireCache& other) = delete;
        ConcurrentExpireCache& operator=(const ConcurrentExpireCache&) = delete;

        ~ConcurrentExpireCache() {
            evict_thread_stop_ = true;
            evict_thread_->join();
            clear();
        }

        /**
         * Find a value by key, and return it by filling the ConstAccessor, which
         * can be default-constructed. Returns true if the element was found, false
         * otherwise.
         */
        bool early_find(ConstAccessor& ac, const TKey& key, int sec);
        bool find_and_postpone(ConstAccessor& ac, const TKey& key, int sec);
        bool find(ConstAccessor& ac, const TKey& key);

        bool early_find(Accessor& ac, const TKey& key, int sec);
        bool find_and_postpone(Accessor& ac, const TKey& key, int sec);
        bool find(Accessor& ac, const TKey& key);

        /**
         * Insert a value into the container. Both the key and value will be copied.
         * The new element will put into the eviction list as the most-recently
         * used.
         *
         * If there was already an element in the container with the same key, it
         * will not be updated, and false will be returned. Otherwise, true will be
         * returned.
         */
        bool insert(const TKey& key, const TValue& value);

        /**
         * Clear the container. NOT THREAD SAFE -- do not use while other threads
         * are accessing the container.
         */
        void clear();

        /**
         * Get the approximate size of the container. May be slightly too low when
         * insertion is in progress.
         */
        size_t size() const {
            return m_size.load();
        }

        /**
         * need to clone map, be careful.
         * @param dest_map
         */
        void getSnapshot(std::unordered_map<TKey, TValue>& dest_map) {
            HashMap clone(m_map);
            for(auto itr = m_map.begin(); itr != m_map.end(); ++itr) {
                dest_map.insert(std::make_pair(itr->first, itr->second.m_value));
            }
        }

    private:
        /**
         * Unlink a node from the list. The caller must lock the list mutex while
         * this is called.
         */
        void delink(ListNode* node);

        /**
         * Add a new node to the list in the most-recently used position. The caller
         * must lock the list mutex while this is called.
         */
        void pushFront(ListNode* node);

        /**
         * Evict the least-recently used item from the container. This function does
         * its own locking.
         */
        void evict();

        void evictTask();

        /**
       * The maximum number of elements in the container.
       */
        size_t m_maxSize;

        /**
         * This atomic variable is used to signal to all threads whether or not
         * eviction should be done on insert. It is approximately equal to the
         * number of elements in the container.
         */
        std::atomic<size_t> m_size;

        /**
         * The underlying TBB hash map.
         */
        HashMap m_map;

        /**
         * The linked list. The "head" is the most-recently used node, and the
         * "tail" is the least-recently used node. The list mutex must be held
         * during both read and write.
         */
        ListNode m_head;
        ListNode m_tail;
        typedef std::mutex ListMutex;
        ListMutex m_listMutex;

        int64_t m_ttlSecond;
        std::shared_ptr<std::thread> evict_thread_;
        int32_t evict_thread_duration_second_;
        std::atomic<bool> evict_thread_stop_;        
    };

    template <class TKey, class TValue, class THash>
    typename ConcurrentExpireCache<TKey, TValue, THash>::ListNode* const
            ConcurrentExpireCache<TKey, TValue, THash>::OutOfListMarker = (ListNode*)-1;

    template <class TKey, class TValue, class THash>
    ConcurrentExpireCache<TKey, TValue, THash>::
    ConcurrentExpireCache(size_t maxSize, int64_t ttlSecond)
            : m_maxSize(maxSize), m_size(0), m_ttlSecond(ttlSecond),
              evict_thread_duration_second_(10), evict_thread_stop_(false),
              m_map(std::thread::hardware_concurrency() * 4) // it will automatically grow
    {
        m_head.m_prev = nullptr;
        m_head.m_next = &m_tail;
        m_tail.m_prev = &m_head;
        evict_thread_.reset(new std::thread(&ConcurrentExpireCache<TKey, TValue, THash>::evictTask, this));
    }

    template <class TKey, class TValue, class THash>
    bool ConcurrentExpireCache<TKey, TValue, THash>::
    early_find(ConstAccessor& ac, const TKey& key, int sec) {
        HashMapConstAccessor& hashAccessor = ac.m_hashAccessor;
        //return m_map.find(hashAccessor, key) && !ac.will_expire(sec);
        if (!m_map.find(hashAccessor, key)) {
            return false;
        }

        if (ac.will_expire(sec, m_ttlSecond)) {
            std::unique_lock<ListMutex> lock(m_listMutex);
            delink(hashAccessor->second.m_listNode);
            lock.unlock();
            m_map.erase(hashAccessor);
            return false;
        }
        return true;
    }

    template <class TKey, class TValue, class THash>
    bool ConcurrentExpireCache<TKey, TValue, THash>::
    find_and_postpone(ConstAccessor& ac, const TKey& key, int sec) {
        HashMapConstAccessor& hashAccessor = ac.m_hashAccessor;
        if (!m_map.find(hashAccessor, key)) {
            return false;
        }

        if(!ac.is_valid()) {
            return false;
        }

        if (ac.will_expire(sec, m_ttlSecond)) {
            //ac.postpone(sec);
            ac.set_invalid();
            m_map.erase(hashAccessor);
            return false;
        }

        return true;
    }

    template <class TKey, class TValue, class THash>
    bool ConcurrentExpireCache<TKey, TValue, THash>::
    find(ConstAccessor& ac, const TKey& key) {
        HashMapConstAccessor& hashAccessor = ac.m_hashAccessor;
        return m_map.find(hashAccessor, key);
    }

    template <class TKey, class TValue, class THash>
    bool ConcurrentExpireCache<TKey, TValue, THash>::
    early_find(Accessor& ac, const TKey& key, int sec) {
        HashMapAccessor& hashAccessor = ac.m_hashAccessor;
        //return m_map.find(hashAccessor, key) && ac.will_expire(sec);
        if (!m_map.find(hashAccessor, key)) {
            return false;
        }

        if (ac.will_expire(sec, m_ttlSecond)) {
            std::unique_lock<ListMutex> lock(m_listMutex);
            delink(hashAccessor->second.m_listNode);
            lock.unlock();
            m_map.erase(hashAccessor);
            return false;
        }
        return true;
    }

    template <class TKey, class TValue, class THash>
    bool ConcurrentExpireCache<TKey, TValue, THash>::
    find_and_postpone(Accessor& ac, const TKey& key, int sec) {
        HashMapAccessor& hashAccessor = ac.m_hashAccessor;
        //return m_map.find(hashAccessor, key) && ac.will_expire(sec);
        if (!m_map.find(hashAccessor, key)) {
            return false;
        }

        if(!ac.is_valid()) {
            return false;
        }

        if (ac.will_expire(sec, m_ttlSecond)) {
            //ac.postpone(sec);
            ac.set_invalid();
            m_map.erase(hashAccessor);
            return false;
        }

        return true;
    }

    template <class TKey, class TValue, class THash>
    bool ConcurrentExpireCache<TKey, TValue, THash>::
    find(Accessor& ac, const TKey& key) {
        HashMapAccessor& hashAccessor = ac.m_hashAccessor;
        return m_map.find(hashAccessor, key);
    }

    template <class TKey, class TValue, class THash>
    bool ConcurrentExpireCache<TKey, TValue, THash>::
    insert(const TKey& key, const TValue& value) {
        if (m_size >= m_maxSize) {
            return false;
        }

        std::time_t timestamp = std::time(NULL);
        ListNode* node = new ListNode(key, timestamp);
        HashMapAccessor hashAccessor;
        HashMapValuePair hashMapValue(key, HashMapValue(value, node));
        if (!m_map.insert(hashAccessor, hashMapValue)) {
            delete node;
            return false;
        }

        // increase before insert
        ++m_size;
        std::unique_lock<ListMutex> lock(m_listMutex);
        pushFront(node);
        lock.unlock();
        return true;
    }

    template <class TKey, class TValue, class THash>
    void ConcurrentExpireCache<TKey, TValue, THash>::
    clear() {
        m_map.clear();
        ListNode* node = m_head.m_next;
        ListNode* next;
        while (node != &m_tail) {
            next = node->m_next;
            delete node;
            node = next;
        }
        m_head.m_next = &m_tail;
        m_tail.m_prev = &m_head;
        m_size = 0;
    }

    template <class TKey, class TValue, class THash>
    inline void ConcurrentExpireCache<TKey, TValue, THash>::
    delink(ListNode* node) {
        ListNode* prev = node->m_prev;
        ListNode* next = node->m_next;
        prev->m_next = next;
        next->m_prev = prev;
        node->m_prev = OutOfListMarker;
    }

    template <class TKey, class TValue, class THash>
    inline void ConcurrentExpireCache<TKey, TValue, THash>::
    pushFront(ListNode* node) {
        ListNode* oldRealHead = m_head.m_next;
        node->m_prev = &m_head;
        node->m_next = oldRealHead;
        oldRealHead->m_prev = node;
        m_head.m_next = node;
    }

    template <class TKey, class TValue, class THash>
    void ConcurrentExpireCache<TKey, TValue, THash>::
    evict() {
        std::vector<ListNode*> expiredNodes;
        std::time_t timestamp = std::time(NULL);
        timestamp -= m_ttlSecond;
        std::unique_lock<ListMutex> lock(m_listMutex);
        ListNode* moribund = m_tail.m_prev;
        while (moribund != &m_head && moribund->m_timestamp < timestamp) {
            delink(moribund);
            expiredNodes.push_back(moribund);
            moribund = m_tail.m_prev;
        }
        lock.unlock();

        if (expiredNodes.empty()) {
            return;
        }
        m_size = m_size - expiredNodes.size();
        for (ListNode* expiredNode : expiredNodes) {
            if(expiredNode->m_timestamp == 0) {
                delete expiredNode;
                continue;
            }
            HashMapAccessor hashAccessor;
            if (!m_map.find(hashAccessor, expiredNode->m_key)) {
                // Presumably unreachable
                continue;
            }
            m_map.erase(hashAccessor);
            delete expiredNode;
        }
    }

    template <class TKey, class TValue, class THash>
    void ConcurrentExpireCache<TKey, TValue, THash>::
    evictTask() {
        while(!evict_thread_stop_) {
            evict();
            std::this_thread::sleep_for(std::chrono::seconds(evict_thread_duration_second_));
        }
    }

}

#endif //ML_PLATFORM_CONCURRENT_EXPIRE_CACHE_H
