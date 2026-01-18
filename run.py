import os
from pathlib import Path
from cocotb.runner import get_runner

rtl_dir = Path(os.getcwd())
my_verilog_files = [rtl_dir / "chipS.v"]


toplevel_module = "neural_net"  
test_module = "testBenchS"        
sim_name = "icarus"
runner = get_runner(sim_name)

print(f"Starting Simulation using {sim_name.upper()}...")

runner.build(
    
    verilog_sources=my_verilog_files,
    hdl_toplevel=toplevel_module,
    always=True, 
    build_dir=rtl_dir / "sim_build" 
)
runner.test(
    hdl_toplevel=toplevel_module,
    test_module=test_module,
    build_dir=rtl_dir / "sim_build"
)