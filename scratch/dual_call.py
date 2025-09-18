import asyncio
from functools import partial
from typing import Callable, Generic, ParamSpec, TypeVar, cast

P = ParamSpec("P")
T = TypeVar("T")


class DualCallable((Generic[P, T])):
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T: ...
    async def async_wait(self, *args: P.args, **kwargs: P.kwargs) -> T: ...

    def __get__(self, instance, owner):
        # Mimic staticmethod: always return self
        return self


def dual_callable(func: Callable[P, T]) -> DualCallable[P, T]:
    func.async_wait = partial(asyncio.to_thread, func)  # type: ignore
    return cast(DualCallable[P, T], func)


class SimpleApi:
    @staticmethod
    @dual_callable
    def get_status(verbose: bool = False) -> str:
        if verbose:
            return "Master is running with full diagnostics"
        return "Master is running"


print(SimpleApi.get_status())
print(asyncio.run(SimpleApi().get_status.async_wait(verbose=True)))
