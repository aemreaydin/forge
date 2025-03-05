use crate::load_image;
use crate::scene::texture::Texture;
use crate::vulkan_context::VulkanContext;
use crate::{load_model, scene::mesh::Mesh};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::{Arc, RwLock},
};
use uuid::Uuid;

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct AssetHandle(Uuid);

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum AssetType {
    Mesh,
    Texture,
}

#[derive(Clone, Debug)]
pub struct AssetMetadata {
    handle: AssetHandle,
    asset_type: AssetType,
    path: PathBuf,
}

pub enum AssetData {
    Mesh(Mesh),
    Texture(Texture),
}

#[derive(Default)]
pub struct AssetRegistry {
    metadatas: RwLock<HashMap<AssetHandle, AssetMetadata>>,
    loaded_meshes: RwLock<HashMap<AssetHandle, Arc<Mesh>>>,
    loaded_textures: RwLock<HashMap<AssetHandle, Arc<Texture>>>,
}

impl AssetRegistry {
    pub fn new() -> Self {
        Self {
            metadatas: RwLock::new(HashMap::new()),
            loaded_meshes: RwLock::new(HashMap::new()),
            loaded_textures: RwLock::new(HashMap::new()),
        }
    }

    pub fn register(
        &self,
        vulkan_context: &VulkanContext,
        path: PathBuf,
        asset_type: AssetType,
    ) -> Option<AssetHandle> {
        if let Some(handle) = self.find_asset_by_path(&path) {
            return Some(handle);
        }

        let handle = AssetHandle(Uuid::new_v4());
        let metadata = AssetMetadata {
            handle,
            path: path.clone(),
            asset_type,
        };

        let mut meta_guard = self
            .metadatas
            .write()
            .expect("Failed to lock metadata for write.");
        meta_guard.insert(handle, metadata);

        match asset_type {
            AssetType::Mesh => tobj::load_obj(
                path,
                &tobj::LoadOptions {
                    triangulate: true,
                    single_index: true,
                    ..Default::default()
                },
            )
            .ok()
            .and_then(|(models, _materials)| models.first().cloned())
            .map(|model| load_model(&model))
            .and_then(|(vertices, indices)| Mesh::new(vulkan_context, vertices, indices).ok())
            .map(|mesh| {
                self.loaded_meshes
                    .write()
                    .expect("Failed to lock mesh assets for write")
                    .insert(handle, Arc::new(mesh));
                handle
            }),
            AssetType::Texture => load_image(path)
                .ok()
                .and_then(|texture_data| Texture::from_2d_data(vulkan_context, texture_data).ok())
                .map(|texture| {
                    self.loaded_textures
                        .write()
                        .expect("Failed to lock mesh assets for write")
                        .insert(handle, Arc::new(texture));
                    handle
                }),
        }
    }

    pub fn get_mesh(&self, handle: AssetHandle) -> Option<Arc<Mesh>> {
        if let Some(metadata) = self.get_metadata(handle) {
            match metadata.asset_type {
                AssetType::Mesh => return self.loaded_meshes.read().unwrap().get(&handle).cloned(),
                AssetType::Texture => return None,
            };
        };
        None
    }

    pub fn get_texture(&self, handle: AssetHandle) -> Option<Arc<Texture>> {
        if let Some(metadata) = self.get_metadata(handle) {
            match metadata.asset_type {
                AssetType::Mesh => return None,
                AssetType::Texture => {
                    return self.loaded_textures.read().unwrap().get(&handle).cloned()
                }
            };
        };
        None
    }

    pub fn unload_assets(&self, vulkan_context: &VulkanContext) {
        self.loaded_meshes
            .read()
            .unwrap()
            .values()
            .for_each(|mesh| mesh.destroy(vulkan_context));
        self.loaded_textures
            .read()
            .unwrap()
            .values()
            .for_each(|texture| texture.destroy(vulkan_context));
    }

    fn get_metadata(&self, handle: AssetHandle) -> Option<AssetMetadata> {
        self.metadatas
            .read()
            .expect("Failed to lock metadatas for read.")
            .get(&handle)
            .cloned()
    }
    fn find_asset_by_path(&self, path: &Path) -> Option<AssetHandle> {
        self.metadatas
            .read()
            .expect("Failed to lock metadata for read")
            .values()
            .find(|val| val.path == path)
            .map(|val| val.handle)
    }
}
