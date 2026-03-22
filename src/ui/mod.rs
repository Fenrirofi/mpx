// ══════════════════════════════════════════════════════════════════════════════
//  src/ui/mod.rs — Panel egui dla Malachite
//
//  Układ:
//    Lewy panel (280px)  → Layer Stack + kontrolki warstwy
//    Prawy panel (300px) → Ustawienia sceny / post-processing
//    Viewport środek     → 3D render (reszta okna)
//
//  Integracja: egui-winit obsługuje eventy, egui-wgpu renderuje na swapchain
//  PO post-processingu (Pass 9 w context.rs).
// ══════════════════════════════════════════════════════════════════════════════
pub mod egui_renderer;

use egui::{self, Color32, RichText, Stroke, Ui};

use crate::layers::{BlendMode, ChannelKind, ChannelValue, Layer, LayerStack};
use crate::renderer::post::PostParams;
use crate::scene::Scene;

// ─────────────────────────────────────────────────────────────────────────────
// UiState — stan całego UI (nie egui Context — ten żyje w EguiRenderer)
// ─────────────────────────────────────────────────────────────────────────────

pub struct UiState {
    // Layer panel
    pub rename_index:    Option<usize>,   // który layer jest edytowany
    pub rename_buf:      String,
    pub show_channels:   bool,            // rozwinięcie sekcji kanałów

    // Settings panel
    pub show_post:       bool,
    pub show_scene:      bool,
    pub show_lights:     bool,
}

impl Default for UiState {
    fn default() -> Self {
        Self {
            rename_index:  None,
            rename_buf:    String::new(),
            show_channels: true,
            show_post:     true,
            show_scene:    false,
            show_lights:   false,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers — kolory motywu
// ─────────────────────────────────────────────────────────────────────────────

const COL_BG:       Color32 = Color32::from_rgb(28,  28,  32 );
const COL_PANEL:    Color32 = Color32::from_rgb(36,  36,  42 );
const COL_ITEM:     Color32 = Color32::from_rgb(46,  46,  54 );
const COL_ACTIVE:   Color32 = Color32::from_rgb(60,  90, 160 );
const COL_ACCENT:   Color32 = Color32::from_rgb(80, 120, 210 );
const COL_TEXT:     Color32 = Color32::from_rgb(220, 220, 228);
const COL_SUBTEXT:  Color32 = Color32::from_rgb(140, 140, 155);
const COL_DANGER:   Color32 = Color32::from_rgb(190,  60,  60 );

/// Konfiguruje ciemny motyw zgodny z Substance Painter
pub fn apply_dark_theme(ctx: &egui::Context) {
    let mut style = (*ctx.style()).clone();

    style.visuals.window_fill          = COL_BG;
    style.visuals.panel_fill           = COL_PANEL;
    style.visuals.faint_bg_color       = COL_ITEM;
    style.visuals.extreme_bg_color     = Color32::from_rgb(18, 18, 22);
    style.visuals.code_bg_color        = COL_ITEM;

    style.visuals.widgets.noninteractive.bg_fill   = COL_ITEM;
    style.visuals.widgets.noninteractive.fg_stroke  = Stroke::new(1.0, COL_SUBTEXT);
    style.visuals.widgets.inactive.bg_fill          = COL_ITEM;
    style.visuals.widgets.inactive.fg_stroke        = Stroke::new(1.0, COL_TEXT);
    style.visuals.widgets.hovered.bg_fill           = Color32::from_rgb(56, 56, 68);
    style.visuals.widgets.hovered.fg_stroke         = Stroke::new(1.0, Color32::WHITE);
    style.visuals.widgets.active.bg_fill            = COL_ACTIVE;
    style.visuals.widgets.active.fg_stroke          = Stroke::new(1.0, Color32::WHITE);
    style.visuals.selection.bg_fill                 = COL_ACTIVE;

    style.visuals.window_corner_radius = egui::CornerRadius::same(4);
    style.visuals.menu_corner_radius   = egui::CornerRadius::same(4);
    style.visuals.window_shadow    = egui::epaint::Shadow::NONE;

    style.spacing.item_spacing     = egui::vec2(6.0, 4.0);
    style.spacing.window_margin    = egui::Margin::same(8);
    style.spacing.button_padding   = egui::vec2(8.0, 4.0);
    style.spacing.slider_rail_height= 4.0;

    ctx.set_style(style);
}

// ─────────────────────────────────────────────────────────────────────────────
// Główna funkcja draw — wywoływana co klatkę z main.rs
// ─────────────────────────────────────────────────────────────────────────────

pub struct UiOutput {
    pub layers_dirty:   bool,  // czy stos warstw się zmienił → odbuduj material
    pub scene_dirty:    bool,  // czy scena się zmieniła → aktualizuj uniforms
    pub viewport_rect:  egui::Rect, // obszar 3D viewportu (bez paneli)
}

pub fn draw(
    ctx:        &egui::Context,
    ui_state:   &mut UiState,
    layers:     &mut LayerStack,
    post_params: &mut PostParams,
    scene:      &mut Scene,
) -> UiOutput {
    let mut layers_dirty = false;
    let mut scene_dirty  = false;

    // ── Lewy panel: Layer Stack ───────────────────────────────────────────────
    egui::SidePanel::left("layers_panel")
        .exact_width(280.0)
        .resizable(false)
        .frame(egui::Frame::none().fill(COL_PANEL).inner_margin(egui::Margin::same(0)))
        .show(ctx, |ui| {
            ui.set_min_height(ui.available_height());

            // Nagłówek panelu
            panel_header(ui, "LAYERS");

            // Pasek narzędzi warstw
            layers_dirty |= layer_toolbar(ui, layers);

            egui::ScrollArea::vertical()
                .id_salt("layers_scroll")
                .show(ui, |ui| {
                    // Lista warstw (odwrócona — góra stosu = góra UI)
                    let n = layers.layers.len();
                    let mut action: Option<LayerAction> = None;

                    for display_i in (0..n).rev() {
                        let is_active = layers.active_index == Some(display_i);
                        if let Some(a) = layer_item(ui, ui_state, &mut layers.layers[display_i],
                                                     display_i, is_active) {
                            action = Some(a);
                        }
                        if is_active { layers_dirty = true; }
                    }

                    if let Some(a) = action {
                        layers_dirty = true;
                        match a {
                            LayerAction::Select(i) => layers.active_index = Some(i),
                            LayerAction::Remove(i) => layers.remove(i),
                            LayerAction::MoveUp(i) => layers.move_up(i),
                            LayerAction::MoveDown(i)=> layers.move_down(i),
                        }
                    }

                    ui.add_space(8.0);

                    // ── Edycja aktywnej warstwy ───────────────────────────────
                    if let Some(idx) = layers.active_index {
                        if idx < layers.layers.len() {
                            ui.separator();
                            section_header(ui, "Active Layer");
                            layers_dirty |= layer_editor(ui, ui_state, &mut layers.layers[idx]);
                        }
                    }
                });
        });

    // ── Prawy panel: Ustawienia ───────────────────────────────────────────────
    egui::SidePanel::right("settings_panel")
        .exact_width(300.0)
        .resizable(false)
        .frame(egui::Frame::none().fill(COL_PANEL).inner_margin(egui::Margin::same(0)))
        .show(ctx, |ui| {
            ui.set_min_height(ui.available_height());
            panel_header(ui, "SETTINGS");

            egui::ScrollArea::vertical()
                .id_salt("settings_scroll")
                .show(ui, |ui| {
                    // Post-processing
                    collapsible(ui, &mut ui_state.show_post, "Post Processing", |ui| {
                        scene_dirty |= post_settings(ui, post_params);
                    });

                    // Scena
                    collapsible(ui, &mut ui_state.show_scene, "Scene Objects", |ui| {
                        scene_dirty |= scene_settings(ui, scene);
                    });

                    // Światła
                    collapsible(ui, &mut ui_state.show_lights, "Lights", |ui| {
                        scene_dirty |= light_settings(ui, scene);
                    });
                });
        });

    // ── Pobierz rect viewportu (środek ekranu między panelami) ───────────────
    let viewport_rect = ctx.available_rect();

    UiOutput { layers_dirty, scene_dirty, viewport_rect }
}

// ─────────────────────────────────────────────────────────────────────────────
// Layer toolbar — przyciski Add / Duplicate
// ─────────────────────────────────────────────────────────────────────────────

enum LayerAction {
    Select(usize),
    Remove(usize),
    MoveUp(usize),
    MoveDown(usize),
}

fn layer_toolbar(ui: &mut Ui, layers: &mut LayerStack) -> bool {
    let mut dirty = false;
    ui.add_space(4.0);
    ui.horizontal(|ui| {
        ui.add_space(8.0);
        if ui.button(RichText::new("＋ Add").color(COL_TEXT)).clicked() {
            let n = layers.layers.len() + 1;
            layers.push(Layer::new(format!("Layer {n}")));
            dirty = true;
        }
        if ui.button(RichText::new("⧉ Dup").color(COL_TEXT)).clicked() {
            if let Some(layer) = layers.active().cloned() {
                let mut new_layer = layer;
                new_layer.name = format!("{} copy", new_layer.name);
                layers.push(new_layer);
                dirty = true;
            }
        }
    });
    ui.add_space(4.0);
    dirty
}

// ─────────────────────────────────────────────────────────────────────────────
// Jeden element listy warstw
// ─────────────────────────────────────────────────────────────────────────────

fn layer_item(
    ui:       &mut Ui,
    _state:   &mut UiState,
    layer:    &mut Layer,
    index:    usize,
    is_active: bool,
) -> Option<LayerAction> {
    let mut action = None;
    let bg = if is_active { COL_ACTIVE } else { COL_ITEM };

    let frame = egui::Frame::none()
        .fill(bg)
        .inner_margin(egui::Margin { left: 8, right: 8, top: 4, bottom: 4 });

    frame.show(ui, |ui| {
        ui.set_min_width(ui.available_width());
        ui.horizontal(|ui| {
            // Widoczność
            let eye = if layer.visible { "👁" } else { "🚫" };
            if ui.small_button(eye).clicked() {
                layer.visible = !layer.visible;
            }

            // Nazwa — kliknięcie = zaznacz
            let response = ui.selectable_label(
                is_active,
                RichText::new(&layer.name).color(COL_TEXT).size(13.0),
            );
            if response.clicked() { action = Some(LayerAction::Select(index)); }
            if response.double_clicked() { action = Some(LayerAction::Select(index)); }

            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                // Usuń
                if ui.small_button(RichText::new("✕").color(COL_DANGER)).clicked() {
                    action = Some(LayerAction::Remove(index));
                }
                // Przesuń
                if ui.small_button("↑").clicked() { action = Some(LayerAction::MoveUp(index)); }
                if ui.small_button("↓").clicked() { action = Some(LayerAction::MoveDown(index)); }
            });
        });

        // Opacity mini-slider pod nazwą
        ui.horizontal(|ui| {
            ui.label(RichText::new("Opacity").color(COL_SUBTEXT).size(11.0));
            let slider = egui::Slider::new(&mut layer.opacity, 0.0..=1.0)
                .show_value(true)
                .fixed_decimals(2);
            ui.add(slider);
        });
    });

    // Cienka linia separatora
    ui.add(egui::Separator::default().spacing(0.0));
    action
}

// ─────────────────────────────────────────────────────────────────────────────
// Edytor aktywnej warstwy — blend mode + kanały
// ─────────────────────────────────────────────────────────────────────────────

fn layer_editor(ui: &mut Ui, state: &mut UiState, layer: &mut Layer) -> bool {
    let mut dirty = false;
    let pad = egui::Margin { left: 8, right: 8, top: 4, bottom: 4 };

    egui::Frame::none().inner_margin(pad).show(ui, |ui| {
        // Nazwa warstwy (edytowalna)
        ui.horizontal(|ui| {
            ui.label(RichText::new("Name").color(COL_SUBTEXT).size(11.0));
            ui.text_edit_singleline(&mut layer.name);
        });

        ui.add_space(4.0);

        // Blend Mode
        ui.horizontal(|ui| {
            ui.label(RichText::new("Blend").color(COL_SUBTEXT).size(11.0));
            egui::ComboBox::from_id_salt("blend_mode")
                .selected_text(layer.blend_mode.label())
                .show_ui(ui, |ui| {
                    for &mode in BlendMode::all() {
                        let selected = layer.blend_mode == mode;
                        if ui.selectable_label(selected, mode.label()).clicked() {
                            layer.blend_mode = mode;
                            dirty = true;
                        }
                    }
                });
        });

        ui.add_space(6.0);

        // Sekcja kanałów — toggle
        ui.horizontal(|ui| {
            let arrow = if state.show_channels { "▾" } else { "▸" };
            if ui.button(RichText::new(format!("{arrow} Channels")).size(12.0)
                           .color(COL_ACCENT)).clicked() {
                state.show_channels = !state.show_channels;
            }
        });

        if state.show_channels {
            ui.add_space(4.0);
            dirty |= channels_editor(ui, layer);
        }
    });

    dirty
}

// ─────────────────────────────────────────────────────────────────────────────
// Kanały w aktywnej warstwie
// ─────────────────────────────────────────────────────────────────────────────

fn channels_editor(ui: &mut Ui, layer: &mut Layer) -> bool {
    let mut dirty = false;

    for ch in &mut layer.channels {
        let frame = egui::Frame::none()
            .fill(if ch.enabled { COL_ITEM } else { Color32::from_rgb(30, 30, 35) })
            .inner_margin(egui::Margin { left: 8, right: 8, top: 6, bottom: 6 })
            .rounding(egui::Rounding::same(3));

        frame.show(ui, |ui| {
            ui.set_min_width(ui.available_width());

            // Nagłówek kanału z toggle
            ui.horizontal(|ui| {
                if ui.checkbox(&mut ch.enabled, "").changed() { dirty = true; }
                ui.label(RichText::new(ch.kind.label())
                    .color(if ch.enabled { COL_TEXT } else { COL_SUBTEXT })
                    .size(12.0)
                    .strong());
            });

            if !ch.enabled { return; }

            ui.add_space(2.0);

            match ch.kind {
                // Color pickers (Vec4)
                ChannelKind::BaseColor | ChannelKind::Emissive => {
                    let mut color = ch.value.as_color();
                    let mut arr = [color.x, color.y, color.z];

                    // Picker RGB
                    if ui.color_edit_button_rgb(&mut arr).changed() {
                        color.x = arr[0]; color.y = arr[1]; color.z = arr[2];
                        ch.value = ChannelValue::Color(color);
                        dirty = true;
                    }

                    if ch.kind == ChannelKind::BaseColor {
                        // Alpha slider dla BaseColor
                        ui.horizontal(|ui| {
                            ui.label(RichText::new("Alpha").color(COL_SUBTEXT).size(11.0));
                            if ui.add(egui::Slider::new(&mut color.w, 0.0..=1.0)
                                         .fixed_decimals(2)).changed() {
                                ch.value = ChannelValue::Color(color);
                                dirty = true;
                            }
                        });
                    } else {
                        // Emissive intensity (w = brightness multiplier)
                        ui.horizontal(|ui| {
                            ui.label(RichText::new("Intensity").color(COL_SUBTEXT).size(11.0));
                            let mut intensity = color.w;
                            if ui.add(egui::Slider::new(&mut intensity, 0.0..=10.0)
                                         .fixed_decimals(2)).changed() {
                                color.w = intensity;
                                ch.value = ChannelValue::Color(color);
                                dirty = true;
                            }
                        });
                    }
                }

                // Scalar sliders
                ChannelKind::Roughness => {
                    let mut v = ch.value.as_scalar();
                    if ui.add(
                        egui::Slider::new(&mut v, 0.0..=1.0)
                            .fixed_decimals(3)
                            .text("roughness")
                    ).changed() {
                        ch.value = ChannelValue::Scalar(v);
                        dirty = true;
                    }
                }
                ChannelKind::Metallic => {
                    let mut v = ch.value.as_scalar();
                    if ui.add(
                        egui::Slider::new(&mut v, 0.0..=1.0)
                            .fixed_decimals(3)
                            .text("metallic")
                    ).changed() {
                        ch.value = ChannelValue::Scalar(v);
                        dirty = true;
                    }
                }
                ChannelKind::Normal => {
                    let mut v = ch.value.as_scalar();
                    if ui.add(
                        egui::Slider::new(&mut v, 0.0..=2.0)
                            .fixed_decimals(2)
                            .text("scale")
                    ).changed() {
                        ch.value = ChannelValue::Scalar(v);
                        dirty = true;
                    }
                }
                ChannelKind::AO => {
                    let mut v = ch.value.as_scalar();
                    if ui.add(
                        egui::Slider::new(&mut v, 0.0..=1.0)
                            .fixed_decimals(3)
                            .text("strength")
                    ).changed() {
                        ch.value = ChannelValue::Scalar(v);
                        dirty = true;
                    }
                }
            }
        });

        ui.add_space(3.0);
    }
    dirty
}

// ─────────────────────────────────────────────────────────────────────────────
// Post-processing settings
// ─────────────────────────────────────────────────────────────────────────────

fn post_settings(ui: &mut Ui, p: &mut PostParams) -> bool {
    let mut dirty = false;
    let pad = egui::Margin { left: 8, right: 8, top: 4, bottom: 4 };
    egui::Frame::none().inner_margin(pad).show(ui, |ui| {
        dirty |= labeled_slider(ui, "Exposure",     &mut p.exposure,          -4.0..=4.0,  2);
        dirty |= labeled_slider(ui, "Bloom",        &mut p.bloom_strength,     0.0..=0.5,  3);
        dirty |= labeled_slider(ui, "SSAO",         &mut p.ssao_strength,      0.0..=1.0,  2);
        dirty |= labeled_slider(ui, "Vignette",     &mut p.vignette_strength,  0.0..=1.0,  2);
        dirty |= labeled_slider(ui, "Chrom. Aber.", &mut p.ca_strength,        0.0..=1.0,  2);
        dirty |= labeled_slider(ui, "Grain",        &mut p.grain_strength,     0.0..=1.0,  2);
        dirty |= labeled_slider(ui, "Contrast",     &mut p.contrast,           0.5..=1.8,  2);
        dirty |= labeled_slider(ui, "Saturation",   &mut p.saturation,         0.0..=1.8,  2);
        dirty |= labeled_slider(ui, "Lift",         &mut p.lift,               -0.1..=0.1, 3);
    });
    dirty
}

// ─────────────────────────────────────────────────────────────────────────────
// Scene object settings — wybór obiektu + podstawowe przesunięcie
// ─────────────────────────────────────────────────────────────────────────────

fn scene_settings(ui: &mut Ui, scene: &mut Scene) -> bool {
    let mut dirty = false;
    let pad = egui::Margin { left: 8, right: 8, top: 4, bottom: 4 };
    egui::Frame::none().inner_margin(pad).show(ui, |ui| {
        for (i, obj) in scene.objects.iter_mut().enumerate() {
            let name = format!("Object {i}");
            egui::CollapsingHeader::new(
                RichText::new(&name).color(COL_TEXT).size(12.0)
            )
            .default_open(i == 0)
            .show(ui, |ui| {
                // Materiał — podstawowe kanały (dla wybranego obiektu w stack)
                ui.horizontal(|ui| {
                    ui.label(RichText::new("Roughness").color(COL_SUBTEXT).size(11.0));
                    if ui.add(egui::Slider::new(&mut obj.material.roughness, 0.0..=1.0)
                                 .fixed_decimals(2)).changed() { dirty = true; }
                });
                ui.horizontal(|ui| {
                    ui.label(RichText::new("Metallic").color(COL_SUBTEXT).size(11.0));
                    if ui.add(egui::Slider::new(&mut obj.material.metallic, 0.0..=1.0)
                                 .fixed_decimals(2)).changed() { dirty = true; }
                });
                // Kolor
                ui.horizontal(|ui| {
                    ui.label(RichText::new("Color").color(COL_SUBTEXT).size(11.0));
                    let mut col = [obj.material.base_color.x, obj.material.base_color.y,
                                   obj.material.base_color.z];
                    if ui.color_edit_button_rgb(&mut col).changed() {
                        obj.material.base_color.x = col[0];
                        obj.material.base_color.y = col[1];
                        obj.material.base_color.z = col[2];
                        dirty = true;
                    }
                });
            });
        }
    });
    dirty
}

// ─────────────────────────────────────────────────────────────────────────────
// Light settings
// ─────────────────────────────────────────────────────────────────────────────

fn light_settings(ui: &mut Ui, scene: &mut Scene) -> bool {
    let mut dirty = false;
    let pad = egui::Margin { left: 8, right: 8, top: 4, bottom: 4 };
    egui::Frame::none().inner_margin(pad).show(ui, |ui| {
        for (i, light) in scene.lights.iter_mut().enumerate() {
            let label = format!("Light {i} ({})", match light.kind {
                crate::scene::light::LightKind::Directional => "sun",
                crate::scene::light::LightKind::Point       => "point",
                crate::scene::light::LightKind::Spot { .. } => "spot",
            });
            egui::CollapsingHeader::new(
                RichText::new(&label).color(COL_TEXT).size(12.0)
            )
            .default_open(false)
            .show(ui, |ui| {
                ui.checkbox(&mut light.enabled, "Enabled");
                ui.horizontal(|ui| {
                    ui.label(RichText::new("Intensity").color(COL_SUBTEXT).size(11.0));
                    if ui.add(egui::Slider::new(&mut light.intensity, 0.0..=50.0)
                                 .fixed_decimals(1)).changed() { dirty = true; }
                });
                ui.horizontal(|ui| {
                    ui.label(RichText::new("Color").color(COL_SUBTEXT).size(11.0));
                    let mut col = [light.color.x, light.color.y, light.color.z];
                    if ui.color_edit_button_rgb(&mut col).changed() {
                        light.color.x = col[0]; light.color.y = col[1]; light.color.z = col[2];
                        dirty = true;
                    }
                });
            });
        }
    });
    dirty
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers UI
// ─────────────────────────────────────────────────────────────────────────────

fn panel_header(ui: &mut Ui, title: &str) {
    let frame = egui::Frame::none()
        .fill(Color32::from_rgb(22, 22, 28))
        .inner_margin(egui::Margin { left: 12, right: 8, top: 8, bottom: 8 });
    frame.show(ui, |ui| {
        ui.set_min_width(ui.available_width());
        ui.label(RichText::new(title)
            .color(COL_ACCENT)
            .size(11.0)
            .extra_letter_spacing(2.0)
            .strong());
    });
    ui.add(egui::Separator::default().spacing(0.0));
}

fn section_header(ui: &mut Ui, title: &str) {
    ui.add_space(4.0);
    ui.label(RichText::new(title).color(COL_ACCENT).size(11.0).strong());
    ui.add_space(2.0);
}

fn collapsible(ui: &mut Ui, open: &mut bool, title: &str, content: impl FnOnce(&mut Ui)) {
    let arrow = if *open { "▾" } else { "▸" };
    let header_frame = egui::Frame::none()
        .fill(Color32::from_rgb(32, 32, 40))
        .inner_margin(egui::Margin { left: 10, right: 8, top: 6, bottom: 6 });

    header_frame.show(ui, |ui| {
        ui.set_min_width(ui.available_width());
        if ui.button(RichText::new(format!("{arrow}  {title}"))
                        .color(COL_TEXT).size(12.0)).clicked() {
            *open = !*open;
        }
    });

    if *open {
        content(ui);
        ui.add_space(4.0);
    }

    ui.add(egui::Separator::default().spacing(0.0));
}

fn labeled_slider(
    ui:    &mut Ui,
    label: &str,
    val:   &mut f32,
    range: std::ops::RangeInclusive<f32>,
    decimals: usize,
) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.add_sized(
            [90.0, 0.0],
            egui::Label::new(RichText::new(label).color(COL_SUBTEXT).size(11.0))
        );
        changed = ui.add(
            egui::Slider::new(val, range).fixed_decimals(decimals)
        ).changed();
    });
    changed
}