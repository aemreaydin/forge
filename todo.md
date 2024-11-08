# Forge Checklist

- [ ] Resizing
- [ ] Syncronization2 research
- [ ] Check queue support first and create queues accordingly.
- [ ] Implement Drop for app and make sure everything is destroyed properly
- [ ] Check out allocations and allocation callbacks
- [ ] Check for format and colorspace when creating a swapchain
- [ ] Pipeline handles might be useful
- [ ] Add pipeline cache
- [ ] Most places have queue family index hard-coded
- [ ] Create own surface without using any libraries (will need to write platform code)
- [ ] Create proper queues for each queue family (compute, transfer, graphics)
- [ ] Check for Vulkan Instance version when running the application
- [ ] Add better check for physical device selection
- [ ] Add pipelineCache for graphics pipeline
- [ ] MSAA
- [ ] Graphics pipeline from JSON or YAML
- [ ] VK_KHR_dynamic_rendering - [Vulkan Samples Link](https://github.com/KhronosGroup/Vulkan-Samples/tree/main/samples/extensions/dynamic_rendering)
- [ ] Run shader files inside rust macros that compile glsl during compile time

! MoltenVK doesn't currently support VK_EXT_shader_object

- [ ] Look at shader objects and VK_EXT_shader_object
- [ ] Use shader objects to render but make sure if its enabled, if not fallback to graphics pipelines.
- [ ] Linked stages in shader_object
