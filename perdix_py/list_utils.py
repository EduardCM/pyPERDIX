from __future__ import annotations

"""List utilities and graph list data structures (Fortran: List.f90)."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ListConn:
    point: tuple[int, int] = (-1, -1)
    next: Optional["ListConn"] = None


@dataclass
class ListJunc:
    n_arm: int = 0
    cnL: list[int] = field(default_factory=list)
    poi_c: int = -1
    next: Optional["ListJunc"] = None


@dataclass
class ListBase:
    id: int = -1
    next: Optional["ListBase"] = None


@dataclass
class ListScaf:
    id: int = -1
    next: Optional["ListScaf"] = None


def List_Insert_Conn(head: Optional[ListConn], elem: ListConn) -> ListConn:
    elem.next = head
    return elem


def List_Insert_Junc(head: Optional[ListJunc], elem: ListJunc) -> ListJunc:
    elem.next = head
    return elem


def List_Insert_Base(head: Optional[ListBase], elem: ListBase) -> ListBase:
    elem.next = head
    return elem


def List_Insert_Scaf(head: Optional[ListScaf], elem: ListScaf) -> ListScaf:
    elem.next = head
    return elem


def List_Delete_Conn(head: Optional[ListConn]) -> None:
    while head is not None:
        nxt = head.next
        head.next = None
        head = nxt


def List_Delete_Junc(head: Optional[ListJunc]) -> None:
    while head is not None:
        nxt = head.next
        head.next = None
        head = nxt


def List_Delete_Base(head: Optional[ListBase]) -> None:
    while head is not None:
        nxt = head.next
        head.next = None
        head = nxt


def List_Delete_Scaf(head: Optional[ListScaf]) -> None:
    while head is not None:
        nxt = head.next
        head.next = None
        head = nxt


def List_Count_Junc(head: Optional[ListJunc]) -> int:
    count = 0
    ptr = head
    while ptr is not None:
        count += 1
        ptr = ptr.next
    return count


def List_Count_Base(head: Optional[ListBase]) -> int:
    count = 0
    ptr = head
    while ptr is not None:
        count += 1
        ptr = ptr.next
    return count


def List_Count_Scaf(head: Optional[ListScaf]) -> int:
    count = 0
    ptr = head
    while ptr is not None:
        count += 1
        ptr = ptr.next
    return count
