import client from "./client";

export const getDossiers = async (params = {}) =>
  (await client.get("/dossiers", { params })).data;
export const getDossier = async (id) =>
  (await client.get(`/dossiers/${id}`)).data;
