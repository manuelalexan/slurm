/*****************************************************************************
 *  list.c
 *****************************************************************************
 *  Copyright (C) 2001-2002 The Regents of the University of California.
 *  Copyright (C) 2021 NVIDIA Corporation.
 *  Produced at Lawrence Livermore National Laboratory (cf, DISCLAIMER).
 *  Written by Chris Dunlap <cdunlap@llnl.gov>.
 *
 *  This file is from LSD-Tools, the LLNL Software Development Toolbox.
 *
 *  LSD-Tools is free software; you can redistribute it and/or modify it under
 *  the terms of the GNU General Public License as published by the Free
 *  Software Foundation; either version 2 of the License, or (at your option)
 *  any later version.
 *
 *  In addition, as a special exception, the copyright holders give permission
 *  to link the code of portions of this program with the OpenSSL library under
 *  certain conditions as described in each individual source file, and
 *  distribute linked combinations including the two. You must obey the GNU
 *  General Public License in all respects for all of the code used other than
 *  OpenSSL. If you modify file(s) with this exception, you may extend this
 *  exception to your version of the file(s), but you are not obligated to do
 *  so. If you do not wish to do so, delete this exception statement from your
 *  version.  If you delete this exception statement from all source files in
 *  the program, then also delete it here.
 *
 *  LSD-Tools is distributed in the hope that it will be useful, but WITHOUT
 *  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 *  FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
 *  more details.
 *
 *  You should have received a copy of the GNU General Public License along
 *  with LSD-Tools; if not, write to the Free Software Foundation, Inc.,
 *  51 Franklin Street, Fifth Floor, Boston, MA 02110-1301  USA.
 *****************************************************************************
 *  Refer to "list.h" for documentation on public functions.
 *****************************************************************************/

#include "config.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "list.h"
#include "log.h"
#include "macros.h"
#include "xassert.h"
#include "xmalloc.h"

/*
** Define slurm-specific aliases for use by plugins, see slurm_xlator.h
** for details.
*/
strong_alias(list_create,	slurm_list_create);
strong_alias(list_destroy,	slurm_list_destroy);
strong_alias(list_is_empty,	slurm_list_is_empty);
strong_alias(list_count,	slurm_list_count);
strong_alias(list_shallow_copy,	slurm_list_shallow_copy);
strong_alias(list_append,	slurm_list_append);
strong_alias(list_append_list,	slurm_list_append_list);
strong_alias(list_transfer,	slurm_list_transfer);
strong_alias(list_transfer_max,	slurm_list_transfer_max);
strong_alias(list_prepend,	slurm_list_prepend);
strong_alias(list_find_first,	slurm_list_find_first);
strong_alias(list_delete_all,	slurm_list_delete_all);
strong_alias(list_delete_ptr,	slurm_list_delete_ptr);
strong_alias(list_for_each,	slurm_list_for_each);
strong_alias(list_for_each_max,	slurm_list_for_each_max);
strong_alias(list_flush,	slurm_list_flush);
strong_alias(list_sort,		slurm_list_sort);
strong_alias(list_flip,		slurm_list_flip);
strong_alias(list_push,		slurm_list_push);
strong_alias(list_pop,		slurm_list_pop);
strong_alias(list_peek,		slurm_list_peek);
strong_alias(list_enqueue,	slurm_list_enqueue);
strong_alias(list_dequeue,	slurm_list_dequeue);
strong_alias(list_iterator_create,	slurm_list_iterator_create);
strong_alias(list_iterator_reset,	slurm_list_iterator_reset);
strong_alias(list_iterator_destroy,	slurm_list_iterator_destroy);
strong_alias(list_next,		slurm_list_next);
strong_alias(list_insert,	slurm_list_insert);
strong_alias(list_find,		slurm_list_find);
strong_alias(list_remove,	slurm_list_remove);
strong_alias(list_delete_item,	slurm_list_delete_item);

/***************
 *  Constants  *
 ***************/
#define LIST_MAGIC 0xDEADBEEF
#define LIST_ITR_MAGIC 0xDEADBEFF
#define LIST_NODE_SIZE sizeof(Node)

#define list_alloc() xmalloc(sizeof(struct xlist))
#define list_realloc(_p, _sz) xrealloc(_p, _sz)
#define list_free(_l) xfree(l)
#define list_array_free(_a) xfree(_a)
#define list_iterator_alloc() xmalloc(sizeof(struct listIterator))
#define list_iterator_free(_i) xfree(_i)

/****************
 *  Data Types  *
 ****************/

typedef void *Node;

struct listIterator {
	unsigned int          magic;        /* sentinel for asserting validity   */
	struct xlist         *list;         /* the list being iterated           */
	unsigned int          pos;          /* the next node to be iterated      */
    unsigned int          prev;         /* the previus node in the iteration */
	struct listIterator  *iNext;        /* iterator chain for list_destroy() */
};

struct xlist {
	unsigned int          magic;        /* sentinel for asserting validity   */
	Node                 *arr;          /* head of the array                 */
	struct listIterator  *iNext;        /* iterator chain for list_destroy() */
	ListDelF              fDel;         /* function to delete node data      */
	unsigned int          size;         /* number of nodes in array          */
	unsigned int          capacity;     /* allocated size of the array       */
	pthread_mutex_t       mutex;        /* mutex to protect access to list   */
};


/****************
 *  Prototypes  *
 ****************/

static void *_list_node_create(List l, unsigned int p, void *x);
static void *_list_node_destroy(List l, unsigned int p);
static void *_list_pop_locked(List l);
static void *_list_append_locked(List l, void *x);
static int _list_grow(List l);
static int _list_reserve(List l, int capacity);

#ifndef NDEBUG
static int _list_mutex_is_locked (pthread_mutex_t *mutex);
#endif

/***************
 *  Functions  *
 ***************/

/* list_create()
 */
List
list_create (ListDelF f)
{
	List l = list_alloc();

	l->magic = LIST_MAGIC;
	l->arr = NULL;
	l->iNext = NULL;
	l->fDel = f;
	l->size = 0;
	l->capacity = 0;
	slurm_mutex_init(&l->mutex);

	return l;
}

/* list_destroy()
 */
void
list_destroy (List l)
{
	ListIterator i, iTmp;
	Node *pp;

	xassert(l != NULL);
	slurm_mutex_lock(&l->mutex);
	xassert(l->magic == LIST_MAGIC);

	i = l->iNext;
	while (i) {
		xassert(i->magic == LIST_ITR_MAGIC);
		i->magic = ~LIST_ITR_MAGIC;
		iTmp = i->iNext;
		list_iterator_free(i);
		i = iTmp;
	}
	pp = l->arr;
	while (((char *)pp) < ((char *)l->arr) + (l->size * LIST_NODE_SIZE)) {
		if (*pp && l->fDel) {
			l->fDel(*pp);
		}
		pp = (Node *)(((char *)pp) + LIST_NODE_SIZE);
	}
	l->magic = ~LIST_MAGIC;
	list_array_free(l->arr);
	slurm_mutex_unlock(&l->mutex);
	slurm_mutex_destroy(&l->mutex);
	list_free(l);
}

/* list_is_empty()
 */
int
list_is_empty (List l)
{
	int n;

	xassert(l != NULL);
	slurm_mutex_lock(&l->mutex);
	xassert(l->magic == LIST_MAGIC);
	n = l->size;
	slurm_mutex_unlock(&l->mutex);

	return (n == 0);
}

/*
 * Return the number of items in list [l].
 * If [l] is NULL, return 0.
 */
int list_count(List l)
{
	int n;

	if (!l)
		return 0;

	slurm_mutex_lock(&l->mutex);
	xassert(l->magic == LIST_MAGIC);
	n = l->size;
	slurm_mutex_unlock(&l->mutex);

	return n;
}

List list_shallow_copy(List l)
{
	List m = list_create(NULL);

	xassert(l != NULL);
	xassert(l->magic == LIST_MAGIC);
	slurm_mutex_lock(&l->mutex);
	slurm_mutex_lock(&m->mutex);

	if (_list_reserve(m, l->size) < 0) {
		return NULL;
	}

	m->size = l->size;
	memcpy((char*) m->arr, (char*) l->arr, l->size * LIST_NODE_SIZE);

	slurm_mutex_unlock(&m->mutex);
	slurm_mutex_unlock(&l->mutex);
	return m;
}

/* list_append()
 */
void *
list_append (List l, void *x)
{
	Node v;

	xassert(l != NULL);
	xassert(x != NULL);
	slurm_mutex_lock(&l->mutex);
	xassert(l->magic == LIST_MAGIC);

	v = _list_append_locked(l, x);
	slurm_mutex_unlock(&l->mutex);

	return v;
}

/* list_append_list()
 */
int
list_append_list (List l, List sub)
{
	ListIterator itr;
	Node v;
	int n = 0;

	xassert(l != NULL);
	xassert(l->fDel == NULL);
	xassert(sub != NULL);
	itr = list_iterator_create(sub);
	while((v = list_next(itr))) {
		if (list_append(l, v))
			n++;
		else
			break;
	}
	list_iterator_destroy(itr);

	return n;
}

/*
 *  Pops off list [sub] to [l] with maximum number of entries.
 *  Set max = 0 to transfer all entries.
 *  Note: list [l] must have the same destroy function as list [sub].
 *  Note: list [sub] may be returned empty, but not destroyed.
 *  Returns a count of the number of items added to list [l].
 */
int list_transfer_max(List l, List sub, int max)
{
	Node v;
	int n = 0;

	xassert(l);
	xassert(sub);
	xassert(l->magic == LIST_MAGIC);
	xassert(sub->magic == LIST_MAGIC);
	xassert(l->fDel == sub->fDel);

	while ((!max || n <= max) && (v = list_pop(sub))) {
		list_append(l, v);
		n++;
	}

	return n;
}

/*
 *  Pops off list [sub] to [l].
 *  Set max = 0 to transfer all entries.
 *  Note: list [l] must have the same destroy function as list [sub].
 *  Note: list [sub] will be returned empty, but not destroyed.
 *  Returns a count of the number of items added to list [l].
 */
int list_transfer(List l, List sub)
{
	return list_transfer_max(l, sub, 0);
}

/* list_prepend()
 */
void *
list_prepend (List l, void *x)
{
	Node v;

	xassert(l != NULL);
	xassert(x != NULL);
	slurm_mutex_lock(&l->mutex);
	xassert(l->magic == LIST_MAGIC);

	v = _list_node_create(l, 0, x);
	slurm_mutex_unlock(&l->mutex);

	return v;
}

/* list_find_first()
 */
void *
list_find_first (List l, ListFindF f, void *key)
{
	Node *p;
	Node v = NULL;

	xassert(l != NULL);
	xassert(f != NULL);
	xassert(key != NULL);
	slurm_mutex_lock(&l->mutex);
	xassert(l->magic == LIST_MAGIC);

	for (p = l->arr; ((char *)p) < ((char *)l->arr) + (l->size * LIST_NODE_SIZE); p = (Node *)(((char *)p) + LIST_NODE_SIZE)) {
		if (f(*p, key)) {
			v = *p;
			break;
		}
	}
	slurm_mutex_unlock(&l->mutex);

	return v;
}

/* list_remove_first()
 */
void *
list_remove_first (List l, ListFindF f, void *key)
{
	int i;
	Node v = NULL;

	xassert(l != NULL);
	xassert(f != NULL);
	xassert(key != NULL);
	slurm_mutex_lock(&l->mutex);
	xassert(l->magic == LIST_MAGIC);

	i = 0;
	while (i < l->size) {
		if (f(*((Node *)(((char *)l->arr) + (i * LIST_NODE_SIZE))), key)) {
			v = _list_node_destroy(l, i);
			break;
		} else {
			i = i + 1;
		}
	}
	slurm_mutex_unlock(&l->mutex);

	return v;
}

/* list_delete_all()
 */
int
list_delete_all (List l, ListFindF f, void *key)
{
	int i;
	Node v;
	int n = 0;

	xassert(l != NULL);
	xassert(f != NULL);
	slurm_mutex_lock(&l->mutex);
	xassert(l->magic == LIST_MAGIC);

	i = 0;
	while (i < l->size) {
		if (f(*(Node*)((((char*)l->arr) + (i * LIST_NODE_SIZE))), key)) {
			if ((v = _list_node_destroy(l, i))) {
				if (l->fDel) {
					l->fDel(v);
				}
				n++;
			}
		}
		else {
			i = i + 1;
		}
	}
	slurm_mutex_unlock(&l->mutex);

	return n;
}

/* list_delete_ptr()
 */
int list_delete_ptr(List l, void *key)
{
	int i;
	Node v;
	int n = 0;

	xassert(l);
	xassert(key);
	slurm_mutex_lock(&l->mutex);
	xassert(l->magic == LIST_MAGIC);

	i = 0;
	while (i < l->size) {
		if (*((Node *)(((char*)l->arr) + (i * LIST_NODE_SIZE))) == key) {
			if ((v = _list_node_destroy(l, i))) {
				if (l->fDel) {
					l->fDel(v);
				}
				n = 1;
				break;
			}
		} else
			i = i + 1;
	}
	slurm_mutex_unlock(&l->mutex);

	return n;
}

/* list_for_each()
 */
int
list_for_each (List l, ListForF f, void *arg)
{
	int max = -1;	/* all values */
	return list_for_each_max(l, &max, f, arg, 1);
}

int list_for_each_nobreak(List l, ListForF f, void *arg)
{
	int max = -1;	/* all values */
	return list_for_each_max(l, &max, f, arg, 0);
}

int list_for_each_max(List l, int *max, ListForF f, void *arg,
		      int break_on_fail)
{
	Node *pp;
	int n = 0;
	bool failed = false;

	xassert(l != NULL);
	xassert(f != NULL);
	slurm_mutex_lock(&l->mutex);
	xassert(l->magic == LIST_MAGIC);

	for (pp = l->arr; (*max == -1 || n < *max) && pp < (Node *)(((char *)l->arr) + l->size * LIST_NODE_SIZE); pp = (Node *)(((char*)pp) + LIST_NODE_SIZE)) {
		n++;
		if (f(*pp, arg) < 0) {
			failed = true;
			if (break_on_fail)
				break;
		}
	}
	*max = l->size - n;
	slurm_mutex_unlock(&l->mutex);

	if (failed)
		n = -n;

	return n;
}

/* list_flush()
 */
int
list_flush (List l)
{
	ListIterator i;
	Node *pp;
	int n = 0;

	xassert(l != NULL);
	slurm_mutex_lock(&l->mutex);
	xassert(l->magic == LIST_MAGIC);

	if (l->fDel) {
		pp = l->arr;
		while (pp < (Node *)(((char *)l->arr) + (l->size * LIST_NODE_SIZE))) {
			l->fDel(*pp);
			pp = (Node *)(((char *)pp) + LIST_NODE_SIZE);
			n++;
		}
	}

	l->size = 0;

	/* Reset all iterators on the list to point
	 * to the head of the list.
	 */
	for (i = l->iNext; i; i = i->iNext) {
		xassert(i->magic == LIST_ITR_MAGIC);
		i->pos = 0;
        i->prev = 0;
	}

	slurm_mutex_unlock(&l->mutex);

	return n;
}

/* list_push()
 */
void *
list_push (List l, void *x)
{
	Node v;

	xassert(l != NULL);
	xassert(x != NULL);
	slurm_mutex_lock(&l->mutex);
	xassert(l->magic == LIST_MAGIC);

	v = _list_node_create(l, 0, x);
	slurm_mutex_unlock(&l->mutex);

	return v;
}

/*
 * Handle translation between ListCmpF and signature required by qsort.
 * glibc has this as __compar_fn_t, but that's non-standard so we define
 * our own instead.
 */
typedef int (*ConstListCmpF) (__const void *, __const void *);

/* list_sort()
 *
 * This function uses the libC qsort().
 *
 */
void
list_sort(List l, ListCmpF f)
{
	ListIterator i;

	xassert(l != NULL);
	xassert(f != NULL);
	xassert(l->magic == LIST_MAGIC);
	slurm_mutex_lock(&l->mutex);

	if (l->size <= 1) {
		slurm_mutex_unlock(&l->mutex);
		return;
	}

	qsort(l->arr, l->size, sizeof(char *), (ConstListCmpF)f);

	/* Reset all iterators on the list to point
	 * to the head of the list.
	 */
	for (i = l->iNext; i; i = i->iNext) {
		xassert(i->magic == LIST_ITR_MAGIC);
		i->pos = 0;
        i->prev = 0;
	}

	slurm_mutex_unlock(&l->mutex);
}

/*
 * list_flip - not called list_reverse due to collision with MariaDB
 */
void list_flip(List l)
{
	Node *pp, *pp_2, temp;
	ListIterator i;
	int index;

	xassert(l);
	xassert(l->magic == LIST_MAGIC);
	slurm_mutex_lock(&l->mutex);

	if (l->size <= 1) {
		slurm_mutex_unlock(&l->mutex);
		return;
	}

	index = 0;
	while (index < l->size/2) {
		pp = ((Node *)(((char *)l->arr) + (index * LIST_NODE_SIZE)));
		pp_2 = ((Node *)(((char *)l->arr) + ((l->size - 1 - index) * LIST_NODE_SIZE)));
		temp = *pp;
		*pp = *pp_2;
		*pp_2 = temp;
		index += 1;
	}

	/*
	 * Reset all iterators on the list to point
	 * to the head of the list.
	 */
	for (i = l->iNext; i; i = i->iNext) {
		xassert(i->magic == LIST_ITR_MAGIC);
		i->pos = 0;
		i->prev = 0;
	}

	slurm_mutex_unlock(&l->mutex);
}

/* list_pop()
 */
void *
list_pop (List l)
{
	Node v;

	xassert(l != NULL);
	slurm_mutex_lock(&l->mutex);
	xassert(l->magic == LIST_MAGIC);

	v = _list_pop_locked(l);
	slurm_mutex_unlock(&l->mutex);

	return v;
}

/* list_peek()
 */
void *
list_peek (List l)
{
	Node v;

	xassert(l != NULL);
	slurm_mutex_lock(&l->mutex);
	xassert(l->magic == LIST_MAGIC);

	v = l->size > 0 ? *l->arr : NULL;	
	slurm_mutex_unlock(&l->mutex);

	return v;
}

/*
 * list_peek_last()
 */
void *list_peek_last(List l)
{
	Node v;

	xassert(l != NULL);
	slurm_mutex_lock(&l->mutex);
	xassert(l->magic == LIST_MAGIC);

	v = l->size > 0 ? *((Node *)(((char*)l->arr) + ((l->size - 1) * LIST_NODE_SIZE))) : NULL;

	slurm_mutex_unlock(&l->mutex);

	return v;
}

/* list_enqueue()
 */
void *
list_enqueue (List l, void *x)
{
	Node v;

	xassert(l != NULL);
	xassert(x != NULL);
	slurm_mutex_lock(&l->mutex);
	xassert(l->magic == LIST_MAGIC);

	v = _list_node_create(l, l->size, x);
	slurm_mutex_unlock(&l->mutex);

	return v;
}

/* list_dequeue()
 */
void *
list_dequeue (List l)
{
	Node v;

	xassert(l != NULL);
	slurm_mutex_lock(&l->mutex);
	xassert(l->magic == LIST_MAGIC);

	v = _list_node_destroy(l, 0);
	slurm_mutex_unlock(&l->mutex);

	return v;
}

/* list_iterator_create()
 */
ListIterator
list_iterator_create (List l)
{
	ListIterator i;

	xassert(l != NULL);
	i = list_iterator_alloc();

	i->magic = LIST_ITR_MAGIC;
	i->list = l;
	slurm_mutex_lock(&l->mutex);
	xassert(l->magic == LIST_MAGIC);

	i->pos = 0;
    i->prev = 0;
	i->iNext = l->iNext;
	l->iNext = i;

	slurm_mutex_unlock(&l->mutex);

	return i;
}

/* list_iterator_reset()
 */
void
list_iterator_reset (ListIterator i)
{
	xassert(i != NULL);
	xassert(i->magic == LIST_ITR_MAGIC);
	slurm_mutex_lock(&i->list->mutex);
	xassert(i->list->magic == LIST_MAGIC);

	i->pos = 0;
    i->prev = 0;

	slurm_mutex_unlock(&i->list->mutex);
}

/* list_iterator_destroy()
 */
void
list_iterator_destroy (ListIterator i)
{
	ListIterator *pi;

	xassert(i != NULL);
	xassert(i->magic == LIST_ITR_MAGIC);
	slurm_mutex_lock(&i->list->mutex);
	xassert(i->list->magic == LIST_MAGIC);

	for (pi = &i->list->iNext; *pi; pi = &(*pi)->iNext) {
		xassert((*pi)->magic == LIST_ITR_MAGIC);
		if (*pi == i) {
			*pi = (*pi)->iNext;
			break;
		}
	}
	slurm_mutex_unlock(&i->list->mutex);

	i->magic = ~LIST_ITR_MAGIC;
	list_iterator_free(i);
}

static void * _list_next_locked(ListIterator i)
{
	Node *p = NULL;

	if (i->prev != i->pos) {
		i->prev = i->prev + 1;
    }
	if ((i->pos < i->list->size)) {         
		p = (Node *)(((char*)i->list->arr) + (i->pos * LIST_NODE_SIZE));
		i->pos = i->pos + 1;
	}

	return (p ? *p : NULL);
}

/* list_next()
 */
void *list_next (ListIterator i)
{
	Node rc;

	xassert(i != NULL);
	xassert(i->magic == LIST_ITR_MAGIC);
	slurm_mutex_lock(&i->list->mutex);
	xassert(i->list->magic == LIST_MAGIC);

	rc = _list_next_locked(i);

	slurm_mutex_unlock(&i->list->mutex);

	return rc;
}

/* list_peek_next()
 */
void *
list_peek_next (ListIterator i)
{
	Node *p;

	xassert(i != NULL);
	xassert(i->magic == LIST_ITR_MAGIC);
	slurm_mutex_lock(&i->list->mutex);
	xassert(i->list->magic == LIST_MAGIC);

	p = (Node *)(((char*)i->list->arr) + (i->pos * LIST_NODE_SIZE));

	return (i->pos < i->list->size ? *p : NULL);
}

/* list_insert()
 */
void *
list_insert (ListIterator i, void *x)
{
	Node v;

	xassert(i != NULL);
	xassert(x != NULL);
	xassert(i->magic == LIST_ITR_MAGIC);
	slurm_mutex_lock(&i->list->mutex);
	xassert(i->list->magic == LIST_MAGIC);

	v = _list_node_create(i->list, i->prev, x);
	slurm_mutex_unlock(&i->list->mutex);

	return v;
}

/* list_find()
 */
void *
list_find (ListIterator i, ListFindF f, void *key)
{
	Node v;

	xassert(i != NULL);
	xassert(f != NULL);
	xassert(key != NULL);
	xassert(i->magic == LIST_ITR_MAGIC);

	slurm_mutex_lock(&i->list->mutex);
	xassert(i->list->magic == LIST_MAGIC);

	while ((v = _list_next_locked(i)) && !f(v, key)) {;}

	slurm_mutex_unlock(&i->list->mutex);

	return v;
}

/* list_remove()
 */
void *
list_remove (ListIterator i)
{
	Node v = NULL;

	xassert(i != NULL);
	xassert(i->magic == LIST_ITR_MAGIC);
	slurm_mutex_lock(&i->list->mutex);
	xassert(i->list->magic == LIST_MAGIC);

    if (i->prev != i->pos)
		v = _list_node_destroy(i->list, i->prev);
	slurm_mutex_unlock(&i->list->mutex);

	return v;
}

/* list_delete_item()
 */
int
list_delete_item (ListIterator i)
{
	Node v;

	xassert(i != NULL);
	xassert(i->magic == LIST_ITR_MAGIC);

	if ((v = list_remove(i))) {
		if (i->list->fDel)
			i->list->fDel(v);
		return 1;
	}

	return 0;
}

/*
* Increases the maximum size of the given list [l].
* Preallocates additional memory space to the list [l] by a factor of one.
* Returns the capacity of the list [l], or -1 if reallocation fails.
* This routine assumes the list is already locked upon entry.
*/
static int _list_grow(List l)
 {
	xassert(l != NULL);
	xassert(l->magic == LIST_MAGIC);
	xassert(_list_mutex_is_locked(&l->mutex));

	if (l->size == l->capacity) {
		l->capacity += l->capacity > 1 ? l->capacity : 1;

		l->arr = list_realloc(l->arr, l->capacity * LIST_NODE_SIZE);
		
		if (l->arr == NULL) {
			return -1;
		}
	}

	l->size = 1 + l->size;

	return l->capacity;
 }

/*
* Preallocates memory space for the given list [l].
* Returns the increased capacity of the list [l], or -1 if reallocation fails.
* This routine assumes the list is already locked upon entry.
*/
int _list_reserve(List l, int n)
{
	xassert(l != NULL);
	xassert(l->magic == LIST_MAGIC);
	xassert(_list_mutex_is_locked(&l->mutex));
	xassert(n != 0);

	if (n <= ((int)l->capacity)) {
		return -1;
	}

	l->arr = list_realloc(l->arr, n * LIST_NODE_SIZE);           	
	
	if (l->arr == NULL) {
		return -1;
	}

	return l->capacity = n;
}

/*
* Inserts data pointed to by [x] into list [l] at index [p],
* Returns a ptr to data [x], or NULL if insertion fails.
* This routine assumes the list is already locked upon entry.
*/
static void *_list_node_create(List l, unsigned int p, void *x)
{
	ListIterator i;

	xassert(l != NULL);
	xassert(l->magic == LIST_MAGIC);
	xassert(_list_mutex_is_locked(&l->mutex));
	xassert(p >= 0 && p <= l->size);  
	xassert(x != NULL);

	if (p < 0 || p > l->size) {
		return NULL;
	}

	if (_list_grow(l) < 0) {
		return NULL;

	}

	Node *pp = (Node *)(((char *)l->arr) + (p * LIST_NODE_SIZE));

	size_t move_size = (((char *)l->arr) + ((l->size - 1) * LIST_NODE_SIZE)) - ((char *)pp);
	memmove(((char*)pp) + LIST_NODE_SIZE, pp, move_size);

	*pp = x;

	for (i = l->iNext; i; i = i->iNext) {
		xassert(i->magic == LIST_ITR_MAGIC);
		if (i->prev == p)
			i->prev = p + 1;
		if (i->pos == p)
			i->pos = p + 1;
	}

	return x;
}

/*
* Removes the node pointed to by index [p] from from list [l],
* Returns the data ptr associated with list item being removed,
* or NULL if [*pp] points to the NULL element.
* This routine assumes the list is already locked upon entry.
*/
static void *_list_node_destroy(List l, unsigned int p)
{
	Node v;
	Node *pp;
	ListIterator i;

	xassert(l != NULL);
	xassert(l->magic == LIST_MAGIC);
	xassert(_list_mutex_is_locked(&l->mutex));
	xassert(p >= 0 && p <= l->size);

	if (p < 0 || p >= l->size) {
		return NULL;
	}

	pp = (Node *)(((char *)l->arr) + p * LIST_NODE_SIZE);

	v = *pp;
	
	size_t move_size = (((char *)l->arr) + l->size * LIST_NODE_SIZE) - (((char *)pp)+ LIST_NODE_SIZE);
	memmove(pp, ((char *)pp) + LIST_NODE_SIZE, move_size);

	l->size = MAX(0, ((int)l->size) - 1);

	for (i = l->iNext; i; i = i->iNext) {
		xassert(i->magic == LIST_ITR_MAGIC);
		if (i->pos == p + 1)
			i->pos = p, i->prev = p;
		else if (i->prev == p + 1)
			i->prev = p;
	}

	return v;
}

#ifndef NDEBUG
static int
_list_mutex_is_locked (pthread_mutex_t *mutex)
{
/*  Returns true if the mutex is locked; o/w, returns false.
 */
	int rc;

	xassert(mutex != NULL);
	rc = pthread_mutex_trylock(mutex);
	return(rc == EBUSY ? 1 : 0);
}
#endif /* !NDEBUG */

/* _list_pop_locked
 *
 * Pop an item from the list assuming the
 * the list is already locked.
 */
static void *
_list_pop_locked(List l)
{
	Node v;

	v = _list_node_destroy(l, 0);

	return v;
}

/* _list_append_locked()
 *
 * Append an item to the list. The function assumes
 * the list is already locked.
 */
static void *
_list_append_locked(List l, void *x)
{
	Node v;

	v = _list_node_create(l, l->size, x);

	return v;
}
