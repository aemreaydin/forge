# Forge Checklist

- [ ] Check ultraviolet crate
- [ ] Use OnceLock for device and instance fns
- [ ] Rust build.rs for shader building
- [x] Resizing
- [ ] Move cmd calls to device with custom builder
- [ ] Depth buffer
- [ ] Descriptor set layout create flags doesn't have PUSH_DESCRIPTOR_BIT_KHR
- [ ] Storage buffers research
- [ ] Synchronization2 research
- [ ] Check queue support first and create queues accordingly.
- [ ] Implement Drop for app and make sure everything is destroyed properly
- [ ] Check out allocations and allocation callbacks
- [ ] Check for format and colorspace when creating a swapchain
- [ ] Pipeline handles might be useful
- [ ] Add pipeline cache
- [ ] Timeline
  semaphores - [Vulkan Samples](https://github.com/KhronosGroup/Vulkan-Samples/tree/main/samples/extensions/timeline_semaphore)
- [ ] Most places have queue family index hard-coded
- [ ] Learn about color blending and use cases
- [ ] Create own surface without using any libraries (will need to write platform code)
- [ ] Add hot-reloading shader functionality
- [ ] Create proper queues for each queue family (compute, transfer, graphics)
- [ ] Check for Vulkan Instance version when running the application
- [ ] Add better check for physical device selection
- [ ] Add pipelineCache for graphics pipeline
- [ ] Check out rust-gpu for shader code
- [ ] Check out bindless descriptors
  in [Vulkan Samples](https://github.com/KhronosGroup/Vulkan-Samples/tree/main/samples/extensions/descriptor_indexing)
- [ ] MSAA
- [ ] Graphics pipeline from JSON or YAML
- [ ]
  VK_EXT_graphics_pipeline_library - [Vulkan Samples](https://github.com/KhronosGroup/Vulkan-Samples/tree/main/samples/extensions/graphics_pipeline_library)
- [ ]
  VK_KHR_dynamic_rendering - [Vulkan Samples](https://github.com/KhronosGroup/Vulkan-Samples/tree/main/samples/extensions/dynamic_rendering)
- [ ] Run shader files inside rust macros that compile glsl during compile time
- [ ] Mesh shaders
- [ ] Validation of createinfos
- [ ] Bounding box for models

! MoltenVK doesn't currently support VK_EXT_shader_object

- [ ] Look at shader objects and VK_EXT_shader_object
- [ ] Use shader objects to render but make sure if its enabled, if not fallback to graphics pipelines.
- [ ] Linked stages in shader_object
