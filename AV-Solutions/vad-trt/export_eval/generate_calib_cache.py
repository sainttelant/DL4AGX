#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Generate INT8 calibration cache for VAD TensorRT engines
# Uses existing demo data for calibration

import os
import sys
import argparse
import numpy as np
from pathlib import Path
import tensorrt as trt
import torch

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class ImageCalibrator(trt.IInt8EntropyCalibrator2):
    """Calibrator for image encoder backbone"""
    
    def __init__(self, calibration_data_dir, cache_file, batch_size=1):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.calibration_data_dir = Path(calibration_data_dir)
        self.cache_file = cache_file
        self.batch_size = batch_size
        self.current_index = 0
        
        # Load all calibration data
        self.calib_data = self._load_calibration_data()
        print(f"[Calibrator] Loaded {len(self.calib_data)} calibration images")
        
    def _load_calibration_data(self):
        """Load calibration data from demo directory"""
        calib_data = []
        
        # Look for img.bin files in frame directories
        if self.calibration_data_dir.is_dir():
            frame_dirs = sorted([d for d in self.calibration_data_dir.iterdir() 
                               if d.is_dir() and d.name.isdigit()])
            
            print(f"[Calibrator] Found {len(frame_dirs)} frame directories")
            
            for frame_dir in frame_dirs[:100]:  # Limit to 100 frames for calibration
                img_bin = frame_dir / "img.bin"
                if img_bin.exists():
                    data = np.fromfile(str(img_bin), dtype=np.float32)
                    calib_data.append(data)
                    
        return calib_data
    
    def get_batch_size(self):
        return self.batch_size
    
    def get_batch(self, names):
        """Return next batch of calibration data"""
        if self.current_index >= len(self.calib_data):
            return None
        
        batch = self.calib_data[self.current_index]
        self.current_index += 1
        
        # Convert numpy array to contiguous array and return pointer
        if isinstance(batch, np.ndarray):
            batch = np.ascontiguousarray(batch)
            return [int(batch.ctypes.data)]
        return None
    
    def read_calibration_cache(self):
        """Read existing calibration cache"""
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'rb') as f:
                return f.read()
        return None
    
    def write_calibration_cache(self, cache):
        """Write calibration cache to file"""
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        with open(self.cache_file, 'wb') as f:
            f.write(cache)
        print(f"[Calibrator] Calibration cache saved to {self.cache_file}")


class HeadCalibrator(trt.IInt8EntropyCalibrator2):
    """Calibrator for head network"""
    
    def __init__(self, calibration_data_dir, cache_file, batch_size=1):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.calibration_data_dir = Path(calibration_data_dir)
        self.cache_file = cache_file
        self.batch_size = batch_size
        self.current_index = 0
        
        # Load all calibration data
        self.calib_data = self._load_calibration_data()
        print(f"[HeadCalibrator] Loaded {len(self.calib_data)} calibration samples")
        
    def _load_calibration_data(self):
        """Load calibration data from demo directory"""
        calib_data = []
        
        if self.calibration_data_dir.is_dir():
            frame_dirs = sorted([d for d in self.calibration_data_dir.iterdir() 
                               if d.is_dir() and d.name.isdigit()])
            
            print(f"[HeadCalibrator] Found {len(frame_dirs)} frame directories")
            
            for frame_dir in frame_dirs[:100]:
                # Load necessary input files for head
                inputs = {}
                
                # Load prev_bev if exists
                prev_bev = frame_dir / "prev_bev.bin"
                if prev_bev.exists():
                    inputs['prev_bev'] = np.fromfile(str(prev_bev), dtype=np.float32)
                
                # Load img_metas
                img_metas_shift = frame_dir / "img_metas.0[shift].bin"
                if img_metas_shift.exists():
                    inputs['shift'] = np.fromfile(str(img_metas_shift), dtype=np.float32)
                
                img_metas_lidar2img = frame_dir / "img_metas.0[lidar2img].bin"
                if img_metas_lidar2img.exists():
                    inputs['lidar2img'] = np.fromfile(str(img_metas_lidar2img), dtype=np.float32)
                
                img_metas_can_bus = frame_dir / "img_metas.0[can_bus].bin"
                if img_metas_can_bus.exists():
                    inputs['can_bus'] = np.fromfile(str(img_metas_can_bus), dtype=np.float32)
                
                if inputs:
                    calib_data.append(inputs)
                    
        return calib_data
    
    def get_batch_size(self):
        return self.batch_size
    
    def get_batch(self, names):
        """Return next batch of calibration data"""
        if self.current_index >= len(self.calib_data):
            return None
        
        batch = self.calib_data[self.current_index]
        self.current_index += 1
        
        # Return data in order matching network inputs
        ptrs = []
        for name in names:
            if name in batch:
                data = np.ascontiguousarray(batch[name])
                ptrs.append(int(data.ctypes.data))
        
        return ptrs if ptrs else None
    
    def read_calibration_cache(self):
        """Read existing calibration cache"""
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'rb') as f:
                return f.read()
        return None
    
    def write_calibration_cache(self, cache):
        """Write calibration cache to file"""
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        with open(self.cache_file, 'wb') as f:
            f.write(cache)
        print(f"[HeadCalibrator] Calibration cache saved to {self.cache_file}")


def build_int8_engine_with_calib(
    onnx_path, 
    output_engine_path, 
    calibrator, 
    plugin_lib=None
):
    """Build INT8 TensorRT engine with calibration"""
    
    print(f"\n[Builder] Building INT8 engine from: {onnx_path}")
    print(f"[Builder] Output engine: {output_engine_path}")
    
    # Load plugins if specified
    if plugin_lib and os.path.exists(plugin_lib):
        print(f"[Builder] Loading plugins from: {plugin_lib}")
        trt.init_libnvinfer_plugins(TRT_LOGGER, "")
        # Load custom plugin library
        import ctypes
        ctypes.CDLL(plugin_lib)
    
    # Create builder
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    
    # Parse ONNX
    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            print("[ERROR] Failed to parse ONNX file")
            for i in range(parser.num_errors):
                print(f"  Error {i}: {parser.get_error(i)}")
            return False
    
    print(f"[Builder] Network parsed successfully")
    print(f"[Builder] Inputs: {[network.get_input(i).name for i in range(network.num_inputs)]}")
    print(f"[Builder] Outputs: {[network.get_output(i).name for i in range(network.num_outputs)]}")
    
    # Create builder config
    config = builder.create_builder_config()
    
    # Enable INT8
    config.set_flag(trt.BuilderFlag.INT8)
    config.int8_calibrator = calibrator
    
    # Performance optimizations
    config.set_flag(trt.BuilderFlag.FP16)  # Also allow FP16 fallback
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    
    # Build engine
    print("[Builder] Building engine (this may take several minutes)...")
    engine = builder.build_serialized_network(network, config)
    
    if engine is None:
        print("[ERROR] Failed to build engine")
        return False
    
    # Save engine
    os.makedirs(os.path.dirname(output_engine_path), exist_ok=True)
    with open(output_engine_path, 'wb') as f:
        f.write(engine)
    
    print(f"[Builder] INT8 engine saved to: {output_engine_path}")
    print(f"[Builder] Engine size: {len(engine) / 1024 / 1024:.2f} MB")
    
    return True


def generate_calibration_cache_with_trtexec(onnx_path, output_cache, output_engine, plugin_lib=None):
    """Generate calibration cache and engine using trtexec"""
    
    import subprocess
    
    print(f"\n[CalibCache] Generating calibration cache using trtexec")
    print(f"  ONNX: {onnx_path}")
    print(f"  Output: {output_engine}")
    
    # Build trtexec command
    cmd = [
        'trtexec',
        '--onnx=' + onnx_path,
        '--int8',
        '--saveEngine=' + output_engine
    ]
    
    # Add plugin library if provided
    if plugin_lib and os.path.exists(plugin_lib):
        cmd.append('--staticPlugins=' + plugin_lib)
        print(f"[CalibCache] Using plugin library: {plugin_lib}")
    
    print(f"[CalibCache] Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 minutes timeout
        )
        
        if result.returncode != 0:
            print(f"[CalibCache] Error: {result.stderr}")
            return False
        
        print(f"[CalibCache] Calibration and engine build completed")
        return True
        
    except subprocess.TimeoutExpired:
        print("[CalibCache] Timeout during calibration")
        return False
    except Exception as e:
        print(f"[CalibCache] Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Generate INT8 calibration cache for VAD engines')
    parser.add_argument('--demo-dir', required=True, 
                       help='Path to demo data directory')
    parser.add_argument('--onnx-dir', required=True,
                       help='Path to ONNX files directory')
    parser.add_argument('--output-dir', default='scratch/calibration',
                       help='Output directory for calibration caches and engines')
    parser.add_argument('--plugin-lib', default='../plugins/build/libplugins.so',
                       help='Path to plugin library (default: ../plugins/build/libplugins.so)')
    parser.add_argument('--build-engine', action='store_true',
                       help='Also build INT8 engines (requires TensorRT Python)')
    parser.add_argument('--generate-cache-only', action='store_true',
                       help='Generate calibration cache without building engines')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("VAD INT8 Calibration Cache Generator")
    print("="*80)
    print(f"Demo data directory: {args.demo_dir}")
    print(f"ONNX files directory: {args.onnx_dir}")
    print(f"Output directory: {args.output_dir}")
    print("="*80)
    
    # Backbone calibration
    print("\n[1/2] Processing Backbone (Image Encoder)")
    backbone_onnx = Path(args.onnx_dir) / "vadv1.extract_img_feat/sim_vadv1.extract_img_feat.onnx"
    backbone_cache = output_dir / "backbone_calib.cache"
    backbone_engine = output_dir / "vadv1.extract_img_feat.int8.engine"
    
    if backbone_onnx.exists():
        print(f"[Backbone] ONNX found: {backbone_onnx}")
        
        if args.generate_cache_only:
            # Use trtexec to build engine (calibration happens during build)
            generate_calibration_cache_with_trtexec(
                str(backbone_onnx),
                str(backbone_cache),
                str(backbone_engine),
                args.plugin_lib
            )
        elif args.build_engine:
            # Create calibrator
            backbone_calibrator = ImageCalibrator(
                args.demo_dir, 
                str(backbone_cache),
                batch_size=1
            )
            
            # Build engine
            build_int8_engine_with_calib(
                str(backbone_onnx),
                str(backbone_engine),
                backbone_calibrator,
                args.plugin_lib
            )
        else:
            # Just print instructions
            print("[Backbone] Note: Use trtexec to build engine with calibration cache")
            print(f"  trtexec --onnx={backbone_onnx} \\")
            print(f"         --int8 --calib={backbone_cache} \\")
            print(f"         --saveEngine={backbone_engine}")
    else:
        print(f"[Backbone] ONNX not found: {backbone_onnx}")
    
    # Head calibration
    print("\n[2/2] Processing Head (Detection Head)")
    head_onnx = Path(args.onnx_dir) / "vadv1_prev.pts_bbox_head.forward/sim_vadv1_prev.pts_bbox_head.forward.onnx"
    head_cache = output_dir / "head_calib.cache"
    head_engine = output_dir / "vadv1_prev.pts_bbox_head.forward.int8.engine"
    
    if head_onnx.exists():
        print(f"[Head] ONNX found: {head_onnx}")
        
        if args.generate_cache_only:
            # Use trtexec to build engine (calibration happens during build)
            generate_calibration_cache_with_trtexec(
                str(head_onnx),
                str(head_cache),
                str(head_engine),
                args.plugin_lib
            )
        elif args.build_engine:
            # Create calibrator
            head_calibrator = HeadCalibrator(
                args.demo_dir,
                str(head_cache),
                batch_size=1
            )
            
            # Build engine
            build_int8_engine_with_calib(
                str(head_onnx),
                str(head_engine),
                head_calibrator,
                args.plugin_lib
            )
        else:
            # Just print instructions
            print("[Head] Note: Use trtexec to build engine with calibration cache")
            print(f"  trtexec --onnx={head_onnx} \\")
            print(f"         --int8 --calib={head_cache} \\")
            print(f"         --saveEngine={head_engine}")
    else:
        print(f"[Head] ONNX not found: {head_onnx}")
    
    print("\n" + "="*80)
    print("Calibration cache generation completed!")
    print("="*80)
    print(f"Calibration caches saved to: {output_dir}")
    print("\nNext steps:")
    print("1. Build INT8 engines using trtexec (see README_wilson.md)")
    print("2. Update config.json to use INT8 engines")
    print("3. Run inference with INT8 engines")


if __name__ == '__main__':
    main()
