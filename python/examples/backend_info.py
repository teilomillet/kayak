from __future__ import annotations

import kayak


def main() -> None:
    print("available_backends:", kayak.available_backends())

    for backend_name in (
        kayak.NUMPY_REFERENCE_BACKEND,
        kayak.MOJO_EXACT_CPU_BACKEND,
    ):
        info = kayak.backend_info(backend_name)
        print(
            backend_name,
            {
                "available": info.available,
                "requires_mojo": info.requires_mojo,
                "query_layouts": info.query_layouts,
                "index_layouts": info.index_layouts,
                "availability_reason": info.availability_reason,
            },
        )


if __name__ == "__main__":
    main()
