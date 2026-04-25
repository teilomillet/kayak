import std.benchmark as benchmark

from std.gpu.host import DeviceContext
from std.sys.defines import get_defined_int


comptime FLOAT32_COUNT = get_defined_int["float32_count", 2048]()
comptime INT8_COUNT = get_defined_int["int8_count", 131072]()
comptime INT64_COUNT = get_defined_int["int64_count", 129]()


def main() raises:
    with DeviceContext() as ctx:
        var f32_device = ctx.enqueue_create_buffer[DType.float32](FLOAT32_COUNT)
        var f32_host_in = ctx.enqueue_create_host_buffer[DType.float32](
            FLOAT32_COUNT
        )
        var f32_host_out = ctx.enqueue_create_host_buffer[DType.float32](
            FLOAT32_COUNT
        )
        for i in range(FLOAT32_COUNT):
            f32_host_in[i] = Float32(i % 17)
            f32_host_out[i] = Float32(0.0)

        def f32_h2d_once() capturing raises:
            f32_device.enqueue_copy_from(f32_host_in)
            ctx.synchronize()

        def f32_d2h_once() capturing raises:
            f32_device.enqueue_copy_to(f32_host_out)
            ctx.synchronize()

        var f32_h2d = benchmark.run[f32_h2d_once](max_iters=3)
        var f32_d2h = benchmark.run[f32_d2h_once](max_iters=3)

        f32_device.enqueue_copy_from(f32_host_in)
        f32_device.enqueue_copy_to(f32_host_out)
        ctx.synchronize()
        var f32_ok = f32_host_out[0] == Float32(0.0) and f32_host_out[
            1
        ] == Float32(1.0)

        var i8_device = ctx.enqueue_create_buffer[DType.int8](INT8_COUNT)
        var i8_host_in = ctx.enqueue_create_host_buffer[DType.int8](INT8_COUNT)
        var i8_host_out = ctx.enqueue_create_host_buffer[DType.int8](INT8_COUNT)
        for i in range(INT8_COUNT):
            i8_host_in[i] = Int8(i % 31)
            i8_host_out[i] = Int8(0)

        def i8_h2d_once() capturing raises:
            i8_device.enqueue_copy_from(i8_host_in)
            ctx.synchronize()

        def i8_d2h_once() capturing raises:
            i8_device.enqueue_copy_to(i8_host_out)
            ctx.synchronize()

        var i8_h2d = benchmark.run[i8_h2d_once](max_iters=3)
        var i8_d2h = benchmark.run[i8_d2h_once](max_iters=3)

        i8_device.enqueue_copy_from(i8_host_in)
        i8_device.enqueue_copy_to(i8_host_out)
        ctx.synchronize()
        var i8_ok = i8_host_out[0] == Int8(0) and i8_host_out[1] == Int8(1)

        var i64_device = ctx.enqueue_create_buffer[DType.int64](INT64_COUNT)
        var i64_host_in = ctx.enqueue_create_host_buffer[DType.int64](
            INT64_COUNT
        )
        var i64_host_out = ctx.enqueue_create_host_buffer[DType.int64](
            INT64_COUNT
        )
        for i in range(INT64_COUNT):
            i64_host_in[i] = Int64(i * 3)
            i64_host_out[i] = Int64(0)

        def i64_h2d_once() capturing raises:
            i64_device.enqueue_copy_from(i64_host_in)
            ctx.synchronize()

        def i64_d2h_once() capturing raises:
            i64_device.enqueue_copy_to(i64_host_out)
            ctx.synchronize()

        var i64_h2d = benchmark.run[i64_h2d_once](max_iters=3)
        var i64_d2h = benchmark.run[i64_d2h_once](max_iters=3)

        i64_device.enqueue_copy_from(i64_host_in)
        i64_device.enqueue_copy_to(i64_host_out)
        ctx.synchronize()
        var i64_ok = i64_host_out[0] == Int64(0) and i64_host_out[1] == Int64(3)

        print("status: ok")
        print("device_name: ", ctx.name())
        print("float32_count: ", FLOAT32_COUNT)
        print("float32_h2d_mean_seconds: ", f32_h2d.mean())
        print("float32_d2h_mean_seconds: ", f32_d2h.mean())
        print("float32_roundtrip_ok: ", f32_ok)
        print("int8_count: ", INT8_COUNT)
        print("int8_h2d_mean_seconds: ", i8_h2d.mean())
        print("int8_d2h_mean_seconds: ", i8_d2h.mean())
        print("int8_roundtrip_ok: ", i8_ok)
        print("int64_count: ", INT64_COUNT)
        print("int64_h2d_mean_seconds: ", i64_h2d.mean())
        print("int64_d2h_mean_seconds: ", i64_d2h.mean())
        print("int64_roundtrip_ok: ", i64_ok)
